from dataclasses import dataclass
from operator import attrgetter

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .binary_trees import BinaryTree
from .binary_trees import BinaryTreeNode
from .learn_rate import Adam
from .normal_dist import NormalDist


def _sigmoid_primitive(activation):
    # The two cases are mathematically equivalent but are separated to improve
    # numerical stability.

    def sigmoid_pos(alpha):
        return 1 / (1 + np.exp(-alpha))

    def sigmoid_neg(alpha):
        exp_alpha = np.exp(alpha)
        return exp_alpha / (1 + exp_alpha)

    return np.piecewise(
        activation,
        condlist=[activation >= 0, activation < 0],
        funclist=[sigmoid_pos, sigmoid_neg]
    )


def _sigmoid_derivative(primitive):
    return primitive * (1 - primitive)


@dataclass(init=False, eq=False)
class _TreeNode(BinaryTreeNode):
    # Assigned training partition
    feature: NDArray
    target: NDArray

    # Distribution parameters
    dist_params: ...

    # This is used to determine from which node to grow the tree. The node with the
    # maximum score drop will be selected.
    score_drop: float

    # Child partitions, if the node can split
    left_feature: NDArray
    left_target: NDArray
    right_feature: NDArray
    right_target: NDArray

    # Splitting criteria
    # Threshold - tau
    # Gain - gamma
    feature_id: int
    tau: NDArray | float
    gamma: NDArray | float

    # Forward prop related
    # Activation level - alpha
    # Degree of truth - rho
    # Firing strength - pi
    alpha: NDArray
    rho: NDArray
    pi: NDArray

    # Backprop related
    # Gradient w.r.t. firing strength - dl_dpi
    # Gradient w.r.t. threshold - dl_dtau
    # Gradient w.r.t. log of gain - dl_dlog_gamma
    dl_dpi: NDArray
    dl_dtau: NDArray
    dl_dlog_gamma: NDArray


class FuzzyProbTree:

    def __init__(self, max_split=10, batch_size=8, epochs=20, min_samples=10,
                 min_score_drop=0, random_seed=None):
        self._min_samples = min_samples
        self._max_split = max_split
        self._min_score_drop = float(min_score_drop)
        self._batch_size = batch_size
        self._epochs = epochs
        self._random = np.random.default_rng(random_seed)

        self._tree = None
        self._dist = NormalDist()

        self._tau = None
        self._gamma = None
        self._log_gamma = None

        self._dl_dtau = None
        self._dl_dlog_gamma = None

        self._tau_lr = Adam()
        self._log_gamma_lr = Adam()

    def _ensure_fitted(self):
        if self._tree is None:
            raise RuntimeError('Model is not fitted.')

    def fit(self, feature, target):
        feature = np.asarray(feature)
        target = np.asarray(target)

        if not feature.ndim == 2:
            raise ValueError('Feature must be 2D having (n_samples, n_features).')

        if not target.ndim == 1:
            raise ValueError('Target must be 1D having shape (n_samples).')

        if len(feature) != len(target):
            raise ValueError('Feature and target contains different number of samples.')

        if np.isnan(feature).any():
            raise ValueError('Feature contains NaN.')

        if np.isnan(target).any():
            raise ValueError('Target contains NaN.')

        feature = feature.transpose()
        target = target.transpose()

        self._build_tree(feature, target)
        self._create_parameters()
        self._dist.create_parameters(self._tree.leaves)
        self._tune_tree(feature, target)

    def predict(self, feature):
        self._ensure_fitted()
        feature = np.asarray(feature)
        if not feature.ndim == 2:
            raise ValueError('Feature must be 2D having (n_samples, n_features).')

        feature = feature.transpose()
        predict = self._forward_prop(feature)
        return predict

    def _build_tree(self, feature, target):
        self._tree = BinaryTree()

        root_node = _TreeNode()
        root_node.feature = feature
        root_node.target = target
        self._evolve_node(root_node)
        self._tree.add_node(root_node)

        splits = tqdm(range(self._max_split), leave=False)
        splits.set_description('Split')

        for _ in splits:
            candidate_leaves = [leaf for leaf in self._tree.leaves if hasattr(leaf, 'score_drop')]
            if not candidate_leaves:
                break

            best_leaf = max(candidate_leaves, key=attrgetter('score_drop'))

            left_child = _TreeNode()
            left_child.feature = best_leaf.left_feature
            left_child.target = best_leaf.left_target

            right_child = _TreeNode()
            right_child.feature = best_leaf.right_feature
            right_child.target = best_leaf.right_target

            self._evolve_node(left_child)
            self._evolve_node(right_child)

            self._tree.add_node(left_child, parent=best_leaf, is_left=True)
            self._tree.add_node(right_child, parent=best_leaf, is_left=False)

    def _evolve_node(self, node: _TreeNode):
        node.dist_params = self._dist.compute_estimate(node.target)
        node.gamma = 1.0
        best_score_drop = self._min_score_drop

        parent_score = self._dist.compute_score(node.dist_params, node.target)

        n_features, n_samples = node.feature.shape
        feature_ids = tqdm(range(n_features), leave=False)
        feature_ids.set_description('Feature')

        for feature_id in feature_ids:
            sort_indices = node.feature[feature_id].argsort()
            feature = node.feature[:, sort_indices]
            target = node.target[sort_indices]

            unique_values = np.unique(feature[feature_id])
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2
            thresholds = tqdm(midpoints, leave=False)
            thresholds.set_description('Threshold')

            for threshold in thresholds:
                split_index = np.searchsorted(feature[feature_id], threshold, side='right')

                if self._min_samples <= split_index <= n_samples - self._min_samples:
                    left_feature = feature[:, :split_index]
                    right_feature = feature[:, split_index:]

                    left_target = target[:split_index]
                    right_target = target[split_index:]

                    left_params = self._dist.compute_estimate(left_target)
                    right_params = self._dist.compute_estimate(right_target)

                    left_score = self._dist.compute_score(left_params, left_target)
                    right_score = self._dist.compute_score(right_params, right_target)

                    score_drop = parent_score - left_score - right_score

                    if score_drop > best_score_drop:
                        best_score_drop = score_drop

                        node.score_drop = score_drop
                        node.feature_id = feature_id
                        node.tau = threshold

                        node.left_feature = left_feature
                        node.right_feature = right_feature

                        node.left_target = left_target
                        node.right_target = right_target

    def _tune_tree(self, feature, target):
        n_samples = len(target)
        n_batches = n_samples // self._batch_size
        n_selection = n_batches * self._batch_size
        losses = np.empty(n_batches)

        epochs = tqdm(range(self._epochs), leave=False)
        epochs.set_description('Epoch')

        for _ in epochs:
            selected_samples = self._random.permutation(range(n_samples))
            selected_samples = selected_samples[:n_selection]

            batch = np.split(selected_samples, n_batches)
            batch = tqdm(batch, leave=False)
            batch.set_description('Batch')

            for index, samples in enumerate(batch):
                batch_feature = feature[:, samples]
                batch_target = target[samples]

                self._forward_prop(batch_feature)
                loss = self._backward_prop(batch_target)

                self._adjust_params()
                self._dist.adjust_params()

                losses[index] = loss

            epochs.set_postfix(min=losses.min(), max=losses.max(), avg=losses.mean())

    def _create_parameters(self):
        self._tau = np.stack([non_leaf.tau for non_leaf in self._tree.non_leaves])
        self._gamma = np.stack([non_leaf.gamma for non_leaf in self._tree.non_leaves])
        self._log_gamma = np.log(self._gamma)

        self._dl_dtau = np.zeros_like(self._tau)
        self._dl_dlog_gamma = np.zeros_like(self._log_gamma)

        for index, non_leaf in enumerate(self._tree.non_leaves):
            non_leaf.tau = self._tau[index, np.newaxis]
            non_leaf.gamma = self._gamma[index, np.newaxis]

            non_leaf.dl_dtau = self._dl_dtau[index, np.newaxis]
            non_leaf.dl_dlog_gamma = self._dl_dlog_gamma[index, np.newaxis]

    def _adjust_params(self):
        self._tau_lr.adjust(param=self._tau, grad=self._dl_dtau)
        self._log_gamma_lr.adjust(param=self._log_gamma, grad=self._dl_dlog_gamma)

        self._gamma[:] = np.exp(self._log_gamma)

    def _forward_prop(self, feature):
        self._tree.root.pi = 1.0

        for node in self._tree.topological_ordering():
            if node not in self._tree.leaves:
                feature_val = feature[node.feature_id]
                node.alpha = -node.gamma * (feature_val - node.tau)
                node.rho = _sigmoid_primitive(activation=node.alpha)

                left_child = self._tree.left_child_of(node)
                right_child = self._tree.right_child_of(node)

                left_child.pi = node.pi * node.rho
                right_child.pi = node.pi * (1 - node.rho)

        prediction = self._dist.forward_prop(self._tree.leaves)
        return prediction

    def _backward_prop(self, target):
        loss = self._dist.backward_prop(target, self._tree.leaves)

        for node in reversed(list(self._tree.topological_ordering())):
            if node in self._tree.non_leaves:
                dpi_l_dpi = node.rho
                dpi_r_dpi = 1 - node.rho

                dl_dpi_l = self._tree.left_child_of(node).dl_dpi
                dl_dpi_r = self._tree.right_child_of(node).dl_dpi
                node.dl_dpi = dl_dpi_l * dpi_l_dpi + dl_dpi_r * dpi_r_dpi

                dpi_l_drho = node.pi
                dpi_r_drho = -node.pi
                dl_drho = dl_dpi_l * dpi_l_drho + dl_dpi_r * dpi_r_drho

                drho_dalpha = _sigmoid_derivative(primitive=node.rho)
                dalpha_dlog_gamma = node.alpha
                dalpha_dtau = node.gamma

                dl_dalpha = dl_drho * drho_dalpha
                dl_dtau = dl_dalpha * dalpha_dtau
                dl_dlog_gamma = dl_dalpha * dalpha_dlog_gamma

                node.dl_dtau[:] = dl_dtau.mean()
                node.dl_dlog_gamma[:] = dl_dlog_gamma.mean()

        return loss

    def plot_tree(self, filename, feature_names=None):
        self._ensure_fitted()

        if feature_names is None:
            feature_names = [f'Feature {n}' for n in range(len(self._tree.root.feature))]

        from pygraphviz import AGraph
        graph = AGraph(directed=True)

        with (np.printoptions(precision=3)):
            for index, node in enumerate(self._tree.non_leaves):
                left_child = self._tree.left_child_of(node)
                right_child = self._tree.right_child_of(node)

                node_label = f'NonLeaf-{index}\nγ={node.gamma}'
                left_label = f'{feature_names[node.feature_id]} ≤ {node.tau}'
                right_label = f'{feature_names[node.feature_id]} > {node.tau}'

                graph.add_node(id(node), label=node_label)
                graph.add_edge(id(node), id(left_child), label=left_label)
                graph.add_edge(id(node), id(right_child), label=right_label)

            for index, node in enumerate(self._tree.leaves):
                node_label = f'Leaf-{index}\n{node.dist_params}'
                graph.add_node(id(node), label=node_label)

        graph.node_attr['shape'] = 'box'
        graph.node_attr['style'] = 'rounded'
        graph.node_attr['fontname'] = 'monospace'
        graph.edge_attr['fontname'] = 'monospace'
        graph.draw(filename, prog='dot')
