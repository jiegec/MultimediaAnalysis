import numpy as np
import os
from scipy.stats import wasserstein_distance


def l2(a, b):
    return np.linalg.norm(a - b, ord=2)

def hi1(a, b):
    return np.sum(np.minimum(a, b))

def hi2(a, b):
    return np.sum(np.minimum(a, b)) / np.sum(b)

def bh(a, b):
    return np.sqrt(1 - np.sum(np.sqrt(a * b)))

def l1(a, b):
    return np.linalg.norm(a - b, ord=1)

def l3(a, b):
    return np.linalg.norm(a - b, ord=3)

def li(a, b):
    return np.linalg.norm(a - b, ord=np.inf)

def ws(a, b):
    return wasserstein_distance(a, b)


dist = bh
bins_count = 128

root = './histo-%d' % bins_count
data = dict()
with open('AllImages.txt', 'r') as f:
    for line in f:
        if len(line.strip()):
            file = line.split(' ')[0]
            data[file] = np.load('%s/%s.histo.npy' % (root, file))


try:
    os.makedirs('ans-%d-%s' % (bins_count, dist.__name__))
except:
    pass
overalls = []
precision = []
with open('QueryImages.txt', 'r') as f:
    for line in f:
        if len(line.strip()):
            file = line.split(' ')[0]
            cate = file.split('/')[0]
            current = np.load('%s/%s.histo.npy' % (root, file))
            file = file.split('.')[0]
            dists = []
            for k in data:
                dists.append((k, dist(current, data[k])))
            dists = list(sorted(dists, key=lambda kv: kv[1]))[1:31]
            correct = 0
            with open('ans-%d-%s/res_%s.txt' % (bins_count, dist.__name__, file.replace('/', '_')), 'w') as output:
                for k, v in dists:
                    if cate == k.split('/')[0]:
                        correct += 1
                    output.write('%s %.3f\n' % (k, v))
            precision.append(correct / 30.0)
            overalls.append('%s %.3f' % (file, correct / 30.0))

with open('ans-%d-%s/res_overall.txt' % (bins_count, dist.__name__), 'w') as output:
    output.write('\n'.join(overalls))
    output.write('\naverage %.3f' % np.average(precision))
