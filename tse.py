__author__ = 'pva701'

import sys

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tse(x_emb, y_true, out_file):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    plt.scatter(x_emb[:, 0], x_emb[:, 1], c=y_true, marker="x")
    plt.savefig(out_file)

vecs = []
labels = []
for l in open(sys.argv[1], 'r'):
    sp = l.split(" ")
    y = int(sp[0])
    labels.append(y)
    x = []
    for xi in sp[1:]:
        x.append(float(xi))
    vecs.append(x)

v_red = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(np.array(vecs))
plot_tse(v_red, labels, sys.argv[2])