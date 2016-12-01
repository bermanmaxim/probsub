######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs
import cvxopt
import numpy as np
from time import time
from pystruct.learners.one_slack_ssvm import NoConstraint
from pystruct.learners import OneSlackSSVM

class OneSlackSSVMHard(OneSlackSSVM):
    """One-slack SSVM with hard-constraints cutting-plane generation
    """

    def __init__(self, model, max_iter=10000, C=1.0, check_constraints=False,
                 verbose=0, negativity_constraint=None, positivity_constraint=None, null_constraints = None,
                 hard_constraints=None, n_jobs=1, break_on_bad=False, 
                 show_loss_every=0, tol=1e-3, inference_cache=0,
                 inactive_threshold=1e-5, inactive_window=50,
                 logger=None, cache_tol='auto', switch_to=None, 
                 generate_hard_constraints=None, initialize_constraints=None, qp_eps=1e-5):

        OneSlackSSVM.__init__(self, model, max_iter, C, check_constraints,
                 verbose, negativity_constraint, positivity_constraint,
                 hard_constraints, n_jobs, break_on_bad, 
                 show_loss_every, tol, inference_cache,
                 inactive_threshold, inactive_window,
                 logger, cache_tol, switch_to)
        if (hard_constraints, positivity_constraint, negativity_constraint, generate_hard_constraints) is (None, None, None, None):
            self.hard_satisfied = True
        else:
            # there are hard constraints to satisfy
            self.hard_satisfied = False
        self.null_constraints = null_constraints
        self.generate_hard_constraints = generate_hard_constraints
        self.initialize_constraints = initialize_constraints
        self.cutting_constraints = []
        self._inference_calls = 0
        self.qp_eps = qp_eps
        self.converged = False
    
#    def generate_hard_constraint(self):
    

    def _get_hard_constraints(self):
        # return combined positivity/negativity/hard constraints
        cstr = self.hard_constraints[:]
        d = self.model.size_joint_feature
        eye = np.identity(d)
        for i in self.positivity_constraint:
            cstr.append((eye[i], 0.))
        for i in self.negativity_constraint:
            cstr.append((-eye[i], 0.))
        return cstr


    def _solve_1_slack_qp(self, constraints, n_samples):
        C = np.float(self.C) * n_samples  # this is how libsvm/svmstruct do it
        joint_features = [c[0] for c in constraints]
        n_soft_constraints = len(joint_features)

        hard_constraints = self._get_hard_constraints()
        joint_features.extend(c[0] for c in hard_constraints)

        n_hard_constraints = len(hard_constraints)
        n_constraints = len(joint_features)

        losses = [c[1] for c in constraints]
        losses.extend(c[1] for c in hard_constraints)

        inneriter = 0
        cutting_constraints = self.cutting_constraints

        while True:
            inneriter += 1
            try:
                n_cutting_constraints = len(cutting_constraints)
                n_all_hard_constraints = n_hard_constraints + n_cutting_constraints

                joint_feature_matrix = np.vstack(joint_features + [c[0] for c in cutting_constraints])
                n_constraints = len(joint_features) + len(cutting_constraints)

                P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T))
                # q contains loss from margin-rescaling
                q = cvxopt.matrix(-np.array((losses + [c[1] for c in cutting_constraints]), dtype=np.float))
                # constraints: all alpha must be >zero
                idy = np.identity(n_constraints)
                
                G = cvxopt.sparse(cvxopt.matrix(-idy))
                h = cvxopt.matrix(np.zeros(n_constraints))

                # equality constraint: sum of all alpha corresponding to soft constraints must be = C
                A = cvxopt.matrix(
                    np.hstack([
                            np.ones((1, n_soft_constraints)), 
                            np.zeros((1, n_all_hard_constraints))
                             ])
                                 )
                b = cvxopt.matrix([C])

                # solve QP model
                cvxopt.solvers.options['feastol'] = self.qp_eps
                try:
                    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
                except ValueError:
                    solution = {'status': 'error'}
                if solution['status'] != "optimal":
                    print("regularizing QP!")
                    P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T)
                                      + 1e-8 * np.eye(joint_feature_matrix.shape[0]))
                    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
                    if solution['status'] != "optimal":
                        raise ValueError("QP solver failed. Try regularizing your QP.")

                # Lagrange multipliers
                a = np.ravel(solution['x'])
                self.w = np.dot(a, joint_feature_matrix)

                if self.generate_hard_constraints is not None:
                    cutting_constraints.append(self.generate_hard_constraints(self))
                else:
                    raise NoConstraint

            except NoConstraint:
                if self.verbose>1 and self.generate_hard_constraints is not None:
                    print("cutting hard constraints satisfied after %d inner loops" % inneriter)
                break

        self.hard_satisfied = True
        
        self.old_solution = solution
        self.prune_constraints(constraints, a[0:n_soft_constraints])

        # Support vectors have non zero lagrange multipliers
        sv = a > self.inactive_threshold * C
        if self.verbose > 1:
            print("%d support vectors out of %d constraints" % (np.sum(sv),
                                                                n_constraints))
        self._all_constraints = n_constraints
        self._n_soft = n_soft_constraints
        self._active_soft = np.sum(sv[:n_soft_constraints])
        self._n_hard = n_hard_constraints
        self._active_hard = np.sum(sv[n_soft_constraints:n_soft_constraints+n_hard_constraints])
        self._n_cutting = n_cutting_constraints
        self._active_cutting = np.sum(sv[n_soft_constraints+n_hard_constraints:])
        if self.verbose > 2:
            print("%d/%d soft; %d/%d hard; %d/%d cutting" % (self._active_soft, n_soft_constraints,
                                                           self._active_hard, n_hard_constraints,
                                                           self._active_cutting, n_cutting_constraints))

        # we needed to flip the sign to make the dual into a minimization
        # model
        return -solution['primal objective']

    def prune_constraints(self, constraints, a):
        # append list for new constraint
        self.alphas.append([])
        assert(len(self.alphas) == len(constraints))
        for constraint, alpha in zip(self.alphas, a):
            constraint.append(alpha)
            constraint = constraint[-self.inactive_window:]

        # prune unused constraints:
        # if the max of alpha in last 50 iterations was small, throw away
        if self.inactive_window != 0:
            max_active = [np.max(constr[-self.inactive_window:])
                          for constr in self.alphas]
            # find strongest constraint that is not ground truth constraint
            strongest = np.max(max_active[1:])
            inactive = np.where(max_active
                                < self.inactive_threshold * strongest)[0]

            for idx in reversed(inactive):
                # if we don't reverse, we'll mess the indices up
                del constraints[idx]
                del self.alphas[idx]

    def _find_new_constraint(self, X, Y, joint_feature_gt, constraints, check=True):
        if self.n_jobs != 1:
            # do inference in parallel
            verbose = max(0, self.verbose - 3)
            Y_hat = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(loss_augmented_inference)(
                    self.model, x, y, self.w, relaxed=True)
                for x, y in zip(X, Y))
        else:
            Y_hat = self.model.batch_loss_augmented_inference(
                X, Y, self.w, relaxed=True)
        # compute the mean over joint_features and losses

        if getattr(self.model, 'rescale_C', False):
            djoint_feature = (joint_feature_gt
                              - self.model.batch_joint_feature(X, Y_hat, Y)) / len(X)
        else:
            djoint_feature = (joint_feature_gt
                              - self.model.batch_joint_feature(X, Y_hat)) / len(X)

        loss_mean = np.mean(self.model.batch_loss(Y, Y_hat))

        violation = loss_mean - np.dot(self.w, djoint_feature)
        if check and self._check_bad_constraint(
                violation, djoint_feature, loss_mean, constraints,
                break_on_bad=self.break_on_bad):
            raise NoConstraint
        self._inference_calls += 1
        return Y_hat, djoint_feature, loss_mean

    def fit(self, X, Y, constraints=None, warm_start=False, initialize=True):
        """Learn parameters using cutting plane method.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : ignored

        warm_start : bool, default=False
            Whether we are warmstarting from a previous fit.

        initialize : boolean, default=True
            Whether to initialize the model for the data (n_states, joint feature size, ...)
            Can be let to True (often no-op if variables are already defined).
        """
        if self.verbose:
            print("Training 1-slack dual structural SVM")
        cvxopt.solvers.options['show_progress'] = self.verbose > 3
        if initialize:
            self.model.initialize(X, Y)
        if self.initialize_constraints is not None:
            self.initialize_constraints(self, X, Y)

        # parse cache_tol parameter
        if self.cache_tol is None or self.cache_tol == 'auto':
            self.cache_tol_ = self.tol
        else:
            self.cache_tol_ = self.cache_tol
        
        self.timestamps_ = [time()]
        if not warm_start:
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.objective_curve_, self.primal_objective_curve_ = [], []
            self.cached_constraint_ = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])
            self.inference_cache_ = None
        elif warm_start == "soft":
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])
        elif warm_start:
            constraints = self.constraints_
        else:
            raise ValueError("warm_start parameter unknown value: %s" % warm_start)

        self.last_slack_ = -1

        # get the joint_feature of the ground truth
        if getattr(self.model, 'rescale_C', False):
            joint_feature_gt = self.model.batch_joint_feature(X, Y, Y)
        else:
            joint_feature_gt = self.model.batch_joint_feature(X, Y)

        # try:
        #     # catch ctrl+c to stop training

        for iteration in range(self.max_iter):
            # main loop
            cached_constraint = False
            if self.verbose > 0:
                print("iteration %d" % iteration)
            if self.verbose > 2:
                print(self)
            try:
                Y_hat, djoint_feature, loss_mean = self._constraint_from_cache(
                    X, Y, joint_feature_gt, constraints)
                cached_constraint = True
            except NoConstraint:
                try:
                    Y_hat, djoint_feature, loss_mean = self._find_new_constraint(
                        X, Y, joint_feature_gt, constraints)
                    self._update_cache(X, Y, Y_hat)
                except NoConstraint:
                    if self.verbose:
                        print("no additional soft constraints")
                    if self.verbose and not self.hard_satisfied:
                        print("solving for remaining hard constraints")
                        continue
                    elif (self.switch_to is not None
                            and self.model.inference_method !=
                            self.switch_to):
                        if self.verbose:
                            print("Switching to %s inference" %
                                  str(self.switch_to))
                        self.model.inference_method_ = \
                            self.model.inference_method
                        self.model.inference_method = self.switch_to
                        continue
                    else:
                        self.converged = True
                        break

            self.timestamps_.append(time() - self.timestamps_[0])
            self._compute_training_loss(X, Y, iteration)
            constraints.append((djoint_feature, loss_mean))

            # compute primal objective
            last_slack = -np.dot(self.w, djoint_feature) + loss_mean
            primal_objective = (self.C * len(X)
                                * max(last_slack, 0)
                                + np.sum(self.w ** 2) / 2)
            self.primal_objective_curve_.append(primal_objective)
            self.cached_constraint_.append(cached_constraint)

            objective = self._solve_1_slack_qp(constraints,
                                               n_samples=len(X))

            # update cache tolerance if cache_tol is auto:
            if self.cache_tol == "auto" and not cached_constraint:
                self.cache_tol_ = (primal_objective - objective) / 4

            self.last_slack_ = np.max([(-np.dot(self.w, djoint_feature) + loss_mean)
                                       for djoint_feature, loss_mean in constraints])
            self.last_slack_ = max(self.last_slack_, 0)

            if self.verbose > 0:
                # the cutting plane objective can also be computed as
                # self.C * len(X) * self.last_slack_ + np.sum(self.w**2)/2
                print("cutting plane objective: %f, primal objective %f"
                      % (objective, primal_objective))
            # we only do this here because we didn't add the gt to the
            # constraints, which makes the dual behave a bit oddly
            self.objective_curve_.append(objective)
            self.constraints_ = constraints
            if self.logger is not None:
                self.logger(self, iteration)

            if self.verbose > 5:
                print(self.w)
        # except KeyboardInterrupt:
        #     if self.verbose > 0:
        #         print("Interrupted by user")
        #     pass
        if self.verbose and self.n_jobs == 1:
            print("calls to inference: %d" % self.model.inference_calls)
        # compute final objective:
        self.timestamps_.append(time() - self.timestamps_[0])
        primal_objective = self._objective(X, Y)
        self.primal_objective_curve_.append(primal_objective)
        self.objective_curve_.append(objective)
        self.cached_constraint_.append(False)

        if self.logger is not None:
            self.logger(self, 'final')

        if self.verbose > 0:
            print("final primal objective: %f gap: %f"
                  % (primal_objective, primal_objective - objective))

        return self
