import numpy as np

from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
from pystruct.models.crf import CRF

from pystruct.inference import inference_dispatch
from pystruct.models.utils import loss_augment_unaries

class EdgeFeaturesLossGraph(EdgeFeatureGraphCRF):
    # class_weight can be 'hamming', 'average' or a custom vector
    def __init__(self, n_states=None, n_features=None, n_edge_features=None,
                 inference_method=None, class_weight='hamming',
                 symmetric_edge_features=None,
                 antisymmetric_edge_features=None,
                 truncate_submodular=False,
                 loss='hamming'):

        EdgeFeatureGraphCRF.__init__(self, n_states, n_features, n_edge_features,
            inference_method, class_weight,
            symmetric_edge_features, antisymmetric_edge_features)
        self.truncate_submodular = truncate_submodular


    def initialize(self, X, Y):
        # Modified from EdgeFeatureGraphCRF and CRF
        n_edge_features = X[0][2].shape[1]
        if self.n_edge_features is None:
            self.n_edge_features = n_edge_features
        elif self.n_edge_features != n_edge_features:
            raise ValueError("Expected %d edge features, got %d"
                             % (self.n_edge_features, n_edge_features))
        # Works for both GridCRF and GraphCRF, but not ChainCRF.
        # funny that ^^
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight(X, Y)

    def loss_augment_unar(self, unary_potentials, y):
        loss_augment_unaries(unary_potentials, np.asarray(y), self.class_weight)

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented Inference for x relative to y using parameters w.

        Finds (approximately)
        armin_y_hat np.dot(w, joint_feature(x, y_hat)) + loss(y, y_hat)
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_features),
            edges are an nd-array of shape (n_edges, 2)

        y : ndarray, shape (n_nodes,)
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape=(size_joint_feature,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(n_nodes)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (n_nodes, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        self.loss_augment_unar(unary_potentials, y)

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def _set_class_weight(self, X=None, Y=None):
        if not hasattr(self, 'size_joint_feature'):
            # we are not initialized yet
            return

        n_states = self.n_labels if hasattr(self, 'n_labels') else self.n_states

        if ((isinstance(self.class_weight, basestring) and self.class_weight == 'hamming')
            or self.class_weight is None):
            self.class_weight = np.ones(n_states)
            self.uniform_class_weight = True
        elif isinstance(self.class_weight, basestring) and self.class_weight == 'average':
            Ns = [0 for s in range(n_states)]
            for y in Y:
                for s in range(n_states):
                    Ns[s] += (y == s).sum()
            N = sum(Ns)
            self.Ns = Ns
            self.N = N
            A = n_states / sum([1./ns for ns in Ns])
            self.class_weight = [A/ns for ns in Ns]
            self.uniform_class_weight = False
        else:
            if len(self.class_weight) != n_states:
                raise ValueError("class_weight must have length n_states or"
                                 " be 'hamming' or 'average'")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False

    def max_loss(self, y):
        # maximum possible loss on y for macro averages
        return np.sum(self.class_weight[y])

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        edge_features = self._get_edge_features(x)
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        energies = np.dot(edge_features, pairwise).reshape(
            edge_features.shape[0], self.n_states, self.n_states)
        if self.truncate_submodular:
            if self.n_states != 2:
                raise NotImplementedError, "truncate_submodular only works with 2 states"
            repulse = energies[:, 0, 1] + energies[:, 1, 0]
            attract = energies[:, 0, 0] + energies[:, 1, 1]
            shift = np.maximum(repulse - attract, 0.) / 4.
            energies[:, 0, 1] += shift
            energies[:, 1, 0] += shift
            energies[:, 0, 0] -= shift
            energies[:, 1, 1] -= shift
        return energies

