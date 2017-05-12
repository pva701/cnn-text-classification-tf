"""
Make trees visualizable in an IPython notebook
"""

try:
    from PIL import ImageFont

    font = ImageFont.core.getfont("/Library/Fonts/Georgia.ttf", 15)


    def text_size(text):
        return max(4, font.getsize(text)[0][0])
except Exception:
    def text_size(text):
        # TODO(contributors): make changes here to incorporate cap and uncap unknown words.
        return max(4, int(len(text) * 1.1))


class SampleDescriptor:
    def __init__(self, words_id, left, right,
                       l_bound, r_bound, labels,
                       euler, euler_l, euler_r):
        self.words = words_id
        self.left = left
        self.right = right
        self.l_bound = l_bound
        self.r_bound = r_bound
        self.labels = labels
        self.euler = euler
        self.euler_l = euler_l
        self.euler_r = euler_r

class LabeledTree(object):
    SCORE_MAPPING = [-12.5, -6.25, 0.0, 6.25, 12.5]

    def __init__(self,
                 depth=0,
                 text=None,
                 label=None,
                 children=None,
                 parent=None,
                 udepth=1):
        self.label = label
        self.children = children if children != None else []
        self.general_children = []
        self.text = text
        self.parent = parent
        self.depth = depth
        self.udepth = udepth
        self.subsent_len = 0
        self.text_list = None
        self.sample = None
        self.is_binary_task = False
        self.exclude_leaves_loss = False

    def set_hyperparameters(self, num_classes, exl_leaves):
        self.num_classes = num_classes
        self.exclude_leaves_loss = exl_leaves

    def uproot(tree):
        """
        Take a subranch of a tree and deep-copy the children
        of this subbranch into a new LabeledTree
        """
        uprooted = tree.copy()
        uprooted.parent = None
        for child in tree.all_children():
            uprooted.add_general_child(child)
        return uprooted

    def shrink_tree(tree, final_depth):
        if tree.udepth <= final_depth:
            return tree
        for branch in tree.general_children:
            if branch.udepth == final_depth:
                return branch.uproot()

    def shrunk_trees(tree, final_depth):
        if tree.udepth <= final_depth:
            yield tree
        for branch in tree.general_children:
            if branch.udepth == final_depth:
                yield branch.uproot()

    def copy(self):
        """
        Deep Copy of a LabeledTree
        """
        return LabeledTree(
            udepth=self.udepth,
            depth=self.depth,
            text=self.text,
            label=self.label,
            children=self.children.copy() if self.children != None else [],
            parent=self.parent)

    def add_child(self, child):
        """
        Adds a branch to the current tree.
        """
        self.children.append(child)
        child.parent = self
        self.udepth = max([child.udepth for child in self.children]) + 1

    def add_general_child(self, child):
        self.general_children.append(child)

    def all_children(self):
        if len(self.children) > 0:
            for child in self.children:
                for subchild in child.all_children():
                    yield subchild
            yield self
        else:
            yield self

    def has_children(self):
        return len(self.children) > 0

    def lowercase(self):
        """
        Lowercase all strings in this tree.
        Works recursively and in-place.
        """
        if len(self.children) > 0:
            for child in self.children:
                child.lowercase()
        else:
            self.text = self.text.lower()
        return self

    def to_words(self):
        if self.text_list is not None:
            return self.text_list
        self.text_list = []

        def rec(node):
            if len(node.children) == 0:
                self.text_list.append(node.text)
            else:
                for child in node.children:
                    rec(child)

        rec(self)
        return self.text_list

    def as_text(self):
        words = self.to_words()
        return " ".join(words)

    def __create_labels(self, node):
        l = [0] * self.num_classes
        l[node.label] = 1
        return l

    def to_sample(self, vocab):
        if self.sample:
            return self.sample

        n = len(self.to_words())
        words_id = [None] * n
        left = []
        right = []
        l_bound = []
        r_bound = []
        labels = [None] * (2 * n - 1)
        euler = []
        euler_l = [None] * (n - 1)
        euler_r = [None] * (n - 1)

        self.list_num = 0
        self.vert_num = n

        def collect_info(node):
            if len(node.children) == 0:
                if node.text in vocab:
                    words_id[self.list_num] = vocab[node.text]
                else:
                    words_id[self.list_num] = 0

                if not self.exclude_leaves_loss:
                    labels[self.list_num] = self.__create_labels(node)

                euler.append(self.list_num)

                self.list_num += 1
                return self.list_num - 1
            else:
                assert len(node.children) == 2
                sv_l = len(euler)
                l_n = collect_info(node.children[0])
                r_n = collect_info(node.children[1])
                euler.append(self.vert_num)
                euler_l[self.vert_num - n] = sv_l
                euler_r[self.vert_num - n] = len(euler)

                if l_n < n:
                    l_bound.append(l_n)
                else:
                    l_bound.append(l_bound[l_n - n])
                if r_n < n:
                    r_bound.append(r_n)
                else:
                    r_bound.append(r_bound[r_n - n])

                left.append(l_n)
                right.append(r_n)

                labels[self.vert_num] = self.__create_labels(node)

                self.vert_num += 1
                return self.vert_num - 1

        collect_info(self)
        assert self.list_num == n
        assert self.vert_num == 2 * n - 1
        assert len(left) == len(right) and len(left) + 1 == len(words_id)

        if self.exclude_leaves_loss:
            labels = labels[n:]
        self.sample = \
            SampleDescriptor(words_id, left, right,
                             l_bound, r_bound, labels,
                             euler, euler_l, euler_r)
        return self.sample

    def __str__(self):
        """
        String representation of a tree as visible in original corpus.

        print(tree)
        #=> '(2 (2 not) (3 good))'

        Outputs
        -------

            str: the String representation of the tree.

        """
        if len(self.children) > 0:
            rep = "(%d " % self.label
            for child in self.children:
                rep += str(child)
            return rep + ")"
        else:
            text = self.text \
                .replace("(", "-LRB-") \
                .replace(")", "-RRB-") \
                .replace("{", "-LCB-") \
                .replace("}", "-RCB-") \
                .replace("[", "-LSB-") \
                .replace("]", "-RSB-")

            return ("(%d %s) " % (self.label, text))

    @staticmethod
    def inject_visualization_javascript(tree_width=1200, tree_height=400, tree_node_radius=10):
        """
        In an Ipython notebook, show SST trees using the same Javascript
        code as used by Jason Chuang's visualisations.
        """
        from .javascript import insert_sentiment_markup
        insert_sentiment_markup(tree_width=tree_width, tree_height=tree_height, tree_node_radius=tree_node_radius)
