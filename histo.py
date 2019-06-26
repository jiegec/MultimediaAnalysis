import numpy
import imageio
import numpy as np
import os

bins_count = 16
#bins_count = 128

def histo(image):
    bins = [0] * bins_count
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            r, g, b = image[i][j]
            if bins_count == 16:
                # 2 : 4 : 2
                bins[(r // 128) + (g // 64) * 2 + (b // 128) * 8] += 1
            elif bins_count == 128:
                # 4 : 8 : 4
                bins[(r // 64) + (g // 32) * 4 + (b // 64) * 32] += 1
            else:
                exit()
    return np.array(bins)

root = './histo-%d' % bins_count
with open('AllImages.txt', 'r') as f:
    for line in f:
        if len(line.strip()):
            file = line.split(' ')[0]
            print(file)
            img = imageio.imread('DataSet/' + file)
            try:
                os.makedirs('%s/%s' % (root, file.split('/')[0]))
            except:
                pass
            np.save('%s/%s.histo' % (root, file), histo(img))