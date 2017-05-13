__author__ = 'pva701'

import pytreebank
import os
from pytreebank import treelstm

def load_sst(sst_path):
    dataset = pytreebank.load_sst(sst_path)
    return ([x.lowercase() for x in dataset["train"]],
            [x.lowercase() for x in dataset["dev"]],
            [x.lowercase() for x in dataset["test"]])

def load_subj_or_mr(dataset_path, pos_bound):
    with open(os.path.join(dataset_path, 'sents.cparents'), 'r') as pars,\
         open(os.path.join(dataset_path, 'sents.toks'), 'r') as toks:
        pairs = []
        for x, y in zip(pars, toks):
            p = list(map(int, x.split(" ")))
            t = y.split(" ")
            words = (len(p) + 1) // 2
            assert words == len(t)
            pairs.append((p, t))

    def read_split(file):
        file_path = os.path.join(dataset_path, file + '.split')
        ret = []
        with open(file_path, 'r') as f:
            s = map(int, f.readline().split(" "))
        for i in s:
            ln = len(pairs[i][0])
            lab = [0] * ln
            for j in range(0, ln):
                if pairs[i][0][j] == 0:
                    lab[j] = int(i < pos_bound)
            ret.append(treelstm.read_tree(pairs[i][0], lab, pairs[i][1]))
        return ret

    return read_split('train'), read_split('dev'), read_split('test')

def load_subj(subj_path):
    return load_subj_or_mr(subj_path, 5000)

def load_mr(mr_path):
    return load_subj_or_mr(mr_path, 5331)

def load_trec(trec_path):
    def load_samples(file):
        classes = {'NUM': 0,
                   'LOC': 1,
                   'DESC': 2,
                   'HUM': 3,
                   'ENTY': 4,
                   'ABBR': 5
                   }
        ret = []
        with open(os.path.join(trec_path, file + '.txt')) as raw,\
             open(os.path.join(trec_path, 'sents_' + file + '.toks')) as toks,\
             open(os.path.join(trec_path, 'sents_' + file + '.cparents')) as pars:
            for r, (t, p) in zip(raw, zip(toks, pars)):
                pos = r.find(':')
                lab = classes[r[:pos]]

                p = list(map(int, p.split(" ")))
                t = t.split(" ")
                words = (len(p) + 1) // 2
                assert words == len(t)
                ln = len(p)
                labs = [0] * ln
                for j in range(0, ln):
                    if p[j] == 0: labs[j] = lab

                ret.append(treelstm.read_tree(p, labs, t))
        return ret

    return load_samples('train'), load_samples('test'), load_samples('test')