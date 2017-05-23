__author__ = 'pva701'

import sys
import matplotlib.pyplot as plt
import os
import numpy as np


def show_algos(algos, out_path):
    plt.rc('font', **{
        'family': 'DejaVu Sans',
        'weight': 'normal'
    })

    plt.figure(figsize=(10, 5))
    plt.xlabel('Длина предложения')
    plt.ylabel('Точность')

    max_x = 0.0
    min_x = 1000
    min_y = 1.0
    for algo in algos:
        for x, y in algo[0]:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)

    labels = []
    algo_names = []
    colors = ['b', 'm']
    markers = ['v', '+']
    l = len(colors)
    c = 0
    for algo in algos:
        xs = []
        ys = []
        for x, y in algo[0]:
            xs.append(x)
            ys.append(y)

        label, = plt.plot(xs, ys,
                          color=colors[c],
                          marker=markers[c],
                          linestyle='-',
                          mew=2.0,
                          mec=colors[c])
        labels.append(label)
        algo_names.append(algo[1])
        c = (c + 1) % l

    min_y = 0.05 * int(min_y / 0.05)
    plt.ylim(ymin=min_y - 0.05, ymax=1.05)
    plt.xlim(xmin=min_x-0.5, xmax=max_x + 1)

    plt.xticks(np.arange(min_x, max_x + 1, 2))
    plt.yticks(np.arange(max(0.0, min_y - 0.05), 1.00, 0.05))

    plt.legend(labels, algo_names, bbox_to_anchor=(0.31, 0.23))
    #plt.show()
    plt.savefig(out_path)


def trim(x):
    while x[-1][1] == 0:
        x.pop()
    while x[0][1] == 0:
        x.pop(0)
    return x

algos = []
out_file = sys.argv[-1]
for file in sys.argv[1:-1]:
    print(file)
    with open(file, 'r') as inp:
        name = os.path.splitext(os.path.basename(file))[0]
        points = []
        for line in inp:
            r = line.split(" ")
            lg = int(r[0])
            tot = int(r[1])
            acc = float(r[2])
            points.append((lg, acc))
        algos.append((trim(points), name))

show_algos(algos, out_file)
