# probsub constraint generation functions #

from __future__ import absolute_import, division, print_function  # We require Python 2.6 or later
import numpy as np
from pystruct.learners.one_slack_ssvm import NoConstraint
import time
from numpy import arange

def get_B(nlabels):
    a = np.eye(nlabels)
    pair_order = [(i, j) for i in range(nlabels) for j in range(nlabels) if i<j]
    B = np.array([np.kron(a[i], a[i])
                  + np.kron(a[j], a[j])
                  - np.kron(a[i], a[j])
                  - np.kron(a[j], a[i]) for (i,j) in pair_order])
    return B, pair_order

def get_B_C3(nlabels):
    a = np.eye(nlabels)
    pair_order = [(i, j) for i in range(nlabels) for j in range(nlabels) if i<j]
    B = np.array([np.kron(a[i], a[i]) for i in range(nlabels)]
                  +
                 [-np.kron(a[i], a[j]) for i in range(nlabels)
                                       for j in range(nlabels) if i != j])
    return B, pair_order

def get_P_t(X_p):
    P = np.vstack(x_p[2] for x_p in X_p).astype(np.float64)
    return P.transpose()

def sparse_mult(A, B, C, mask) :
    """
    Computes [A * B][mask] and stores the result in C[mask]
    """
    nrow, ncol = C.shape
    # Computes the product A*B on mask and stores the result in C
    for i in range(nrow):
        for j in range(ncol):
            if mask[i, j]:
                C[i, j] = np.dot(A[i, :], B[:, j])
    return C

def update_L(ssvm, w_pair):
    # compute lower bounds on the violation
    # and only compute negative terms exactly
    diff = w_pair.flatten() - ssvm.last_w_pair.flatten()
    ssvm.L = ssvm.L - np.sqrt(diff.dot(diff)) * ssvm.norm_cstr
    ssvm.L[ssvm.added>0] = 0.
    calcmask = ssvm.L < 0.
    Bw = ssvm.B.dot(w_pair)
    sparse_mult(Bw, ssvm.P_t, ssvm.L, calcmask)

    elmnts = calcmask.sum()
    ssvm.hist_neg.append(elmnts)
    ssvm.hist_pos.append(calcmask.size - elmnts)
    ssvm.hist_qpits.append(len(ssvm.objective_curve_)) 

def update_L_slow(ssvm, w_pair): # full computation
    calcmask = np.ones(ssvm.L.shape)
    Bw = ssvm.B.dot(w_pair)
    sparse_mult(Bw, ssvm.P_t, ssvm.L, calcmask)

    elmnts = calcmask.sum()
    ssvm.hist_neg.append(elmnts)
    ssvm.hist_pos.append(calcmask.size - elmnts)
    ssvm.hist_qpits.append(len(ssvm.objective_curve_))    



def get_helpers(params, cdict):
    if cdict['type'] not in ['C3', 'C4']:
        return (None, None)
    params_prob = params['probsub']

    def initialize_probsub(ssvm, X, Y):
        if cdict['type'] == 'C3':
            B, ssvm.pair_order = get_B_C3(ssvm.model.n_states)
        elif cdict['type'] == 'C4':
            B, ssvm.pair_order = get_B(ssvm.model.n_states)

        ssvm.B = B
        P_t = get_P_t([ex['x'] for ex in cdict['examples']])
        ssvm.P_t = P_t
        
        ssvm.L = np.full([ssvm.B.shape[0], ssvm.P_t.shape[1]], -np.inf) # lower bound on cutting constraints
        ssvm.last_w_pair = np.zeros((ssvm.B.shape[1], ssvm.P_t.shape[0]))
        # if params_prob['compute'] == 'delayed':
        # norms of constraints (for lower-bounding violation)
        ssvm.norm_cstr = np.sqrt((B**2).dot(np.ones((B.shape[1], P_t.shape[0]))).dot(P_t**2))
        
        # debugging: added constraints and iteration
        ssvm.constraintcalls = 0
        #ssvm.added = np.full((ssvm.B.shape[0], ssvm.P_t.shape[1]), False, dtype=np.int)
        ssvm.added = np.zeros((ssvm.B.shape[0], ssvm.P_t.shape[1]), dtype=np.int)
        ssvm.hist_neg = []
        ssvm.hist_pos = []
        ssvm.hist_qpits = []
        ssvm.cut_time = []
        #

    def new_probsub_constraint(ssvm):
        ssvm.cut_time.append(time.time())
        ssvm.constraintcalls += 1
        # #
        # global totalcuttime
        # #
        m = ssvm.model
        
        w_pair = ssvm.w[m.n_states * m.n_features:].reshape(m.n_states * m.n_states, m.n_edge_features)
        # t = time.time()
        if params_prob['compute'] == 'full':
            ssvm.L = ssvm.B.dot(w_pair.dot(ssvm.P_t))
        elif params_prob['compute'] == 'slow':
            update_L_slow(ssvm, w_pair)
        elif params_prob['compute'] == 'delayed':
            update_L(ssvm, w_pair)
        # totalcuttime += time.time() - t
        ssvm.last_w_pair = w_pair
        
        min_idx_pair, min_idx_P = np.unravel_index(ssvm.L.argmin(), ssvm.L.shape)
        if ssvm.L[min_idx_pair, min_idx_P] < -ssvm.qp_eps:
            constraint = np.hstack( (np.zeros(ssvm.model.n_states * ssvm.model.n_features), 
                                     np.kron(ssvm.B[min_idx_pair, :], ssvm.P_t[:, min_idx_P].transpose())) )        
            if ssvm.verbose > 2:
                print("adding probsub constraint (%d, %d), violation %E" % (min_idx_P, min_idx_pair, ssvm.L[min_idx_pair, min_idx_P]))
            # debug: keep track of added constraints
            assert ssvm.added[min_idx_pair, min_idx_P] == 0, "constraint already added"
            ssvm.added[min_idx_pair, min_idx_P] = ssvm.constraintcalls
            #
            # totalcuttime += time.time() - t
            ssvm.cut_time[-1] = (time.time() - ssvm.cut_time[-1])
            return (constraint, 0.)
        else:
            ssvm.cut_time[-1] = (time.time() - ssvm.cut_time[-1])
            raise NoConstraint
    return (initialize_probsub, new_probsub_constraint)
    #                               #











