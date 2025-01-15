from collections import deque


class BinaryTree:
    __slots__ = (
        '_root',
        '_nodes',
        '_leaves',
        '_non_leaves',
        '_parents',
        '_left_children',
        '_right_children',
    )

    def __init__(self):
        self._root = None
        self._nodes = list()
        self._leaves = list()
        self._non_leaves = list()
        self._parents = dict()
        self._left_children = dict()
        self._right_children = dict()

    def add_node(self, node, parent=None, is_left=True):

        if node in self._nodes:
            raise ValueError(f'{self} contains {node}.')

        if parent is None:
            if self._root is not None:
                raise ValueError(f'Root exists.')

            self._root = node

        else:
            if parent not in self._nodes:
                raise ValueError(f'Parent {parent} is not in {self}.')

            if parent in self._leaves:
                self._leaves.remove(parent)
                self._non_leaves.append(parent)

            if is_left:
                if self._left_children[parent] is not None:
                    raise ValueError(f'Left child of parent {parent} exists.')

                self._left_children[parent] = node

            else:
                if self._right_children[parent] is not None:
                    raise ValueError(f'Right child of parent {parent} exists.')

                self._right_children[parent] = node

        self._nodes.append(node)
        self._leaves.append(node)
        self._parents[node] = parent
        self._left_children[node] = None
        self._right_children[node] = None

    @property
    def root(self):
        return self._root

    @property
    def nodes(self):
        return self._nodes

    @property
    def leaves(self):
        return self._leaves

    @property
    def non_leaves(self):
        return self._non_leaves

    def __contains__(self, node):
        return node in self._nodes

    def __len__(self):
        return len(self.nodes)

    def parent_of(self, node):
        if node not in self:
            raise LookupError(f'{node} is not in {self}.')

        return self._parents[node]

    def left_child_of(self, node):
        if node not in self:
            raise LookupError(f'{node} is not in {self}.')

        return self._left_children[node]

    def right_child_of(self, node):
        if node not in self:
            raise LookupError(f'{node} is not in {self}.')

        return self._right_children[node]

    def ancestors_of(self, node):
        if node not in self:
            raise LookupError(f'{node} is not in {self}.')

        ancestor = self.parent_of(node)

        while ancestor is not None:
            yield ancestor
            ancestor = self.parent_of(ancestor)

    def descendants_of(self, node):
        """
        Returns descendants in level-order traversal: root, left, right, ...
        """
        if node not in self:
            raise LookupError(f'{node} is not in {self}.')

        descendants = deque()
        descendants.append(self.left_child_of(node))
        descendants.append(self.right_child_of(node))

        while descendants:
            descendant = descendants.popleft()

            if descendant is not None:
                yield descendant
                descendants.append(self.left_child_of(descendant))
                descendants.append(self.right_child_of(descendant))

    def topological_ordering(self):
        if self.root is not None:
            yield self.root
            yield from self.descendants_of(self.root)


class BinaryTreeNode:
    __slots__ = ()
