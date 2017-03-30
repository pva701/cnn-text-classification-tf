"""
Make trees visualizable in an IPython notebook
"""
import json

try:
    from PIL import ImageFont

    font = ImageFont.core.getfont("/Library/Fonts/Georgia.ttf", 15)


    def text_size(text):
        return max(4, font.getsize(text)[0][0])
except Exception:
    def text_size(text):
        # TODO(contributors): make changes here to incorporate cap and uncap unknown words.
        return max(4, int(len(text) * 1.1))


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
        self.text_list = None
        self.sample = None

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

    def to_sample(self, vocab):
        if self.sample is not None:
            return self.sample

        n = len(self.to_words())
        words_id = [None]*n
        left = []
        right = []
        labels = [None] * (2 * n - 1)
        self.list_num = 0
        self.vert_num = n

        def rec(node):
            if len(node.children) == 0:
                if node.text in vocab:
                    words_id[self.list_num] = vocab[node.text]
                else:
                    words_id[self.list_num] = 0

                l = [0]*5
                l[node.label] = 1
                labels[self.list_num] = l

                self.list_num += 1
                return self.list_num - 1
            else:
                assert len(node.children) == 2
                l_n = rec(node.children[0])
                r_n = rec(node.children[1])
                left.append(l_n)
                right.append(r_n)

                l = [0]*5
                l[node.label] = 1
                labels[self.vert_num] = l

                self.vert_num += 1
                return self.vert_num - 1
        rec(self)
        assert self.list_num == n
        assert self.vert_num == 2 * n - 1
        assert len(left) == len(right) and len(left) + 1 == len(words_id)
        self.sample = (words_id, left, right, labels)
        return self.sample


    def to_dict(self, index=0):
        """
        Dict format for use in Javascript / Jason Chuang's display technology.
        """
        index += 1
        rep = {}
        rep["index"] = index
        rep["leaf"] = len(self.children) == 0
        rep["depth"] = self.udepth
        rep["scoreDistr"] = [0.0] * len(LabeledTree.SCORE_MAPPING)
        # dirac distribution at correct label
        if self.label is not None:
            rep["scoreDistr"][self.label] = 1.0
            mapping = LabeledTree.SCORE_MAPPING[:]
            rep["rating"] = mapping[self.label] - min(mapping)
        # if you are using this method for printing predictions
        # from a model, the the dot product with the model's output
        # distribution should be taken with this list:
        rep["numChildren"] = len(self.children)
        text = self.text if self.text != None else ""
        seen_tokens = 0
        witnessed_pixels = 0
        for i, child in enumerate(self.children):
            if i > 0:
                text += " "
            child_key = "child%d" % (i)
            (rep[child_key], index) = child.to_dict(index)
            text += rep[child_key]["text"]
            seen_tokens += rep[child_key]["tokens"]
            witnessed_pixels += rep[child_key]["pixels"]

        rep["text"] = text
        rep["tokens"] = 1 if (self.text != None and len(self.text) > 0) else seen_tokens
        rep["pixels"] = witnessed_pixels + 3 if len(self.children) > 0 else text_size(self.text)
        return (rep, index)

    def to_json(self):
        rep, _ = self.to_dict()
        return json.dumps(rep)

    def display(self):
        from IPython.display import Javascript, display

        display(Javascript("createTrees([" + self.to_json() + "])"))
        display(Javascript("updateTrees()"))

    def to_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_lines(), self.children[1].to_lines()
            self_line = [left_lines[0] + " " + right_lines[0]]
            return self_line + left_lines + right_lines
        else:
            return [self.text]

    def to_labeled_lines(self):
        if len(self.children) > 0:
            left_lines, right_lines = self.children[0].to_labeled_lines(), self.children[1].to_labeled_lines()
            self_line = [(self.label, left_lines[0][1] + " " + right_lines[0][1])]
            return self_line + left_lines + right_lines
        else:
            return [(self.label, self.text)]

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
