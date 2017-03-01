import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import re

parser = argparse.ArgumentParser()

parser.add_argument("--folder", type=str)
args = parser.parse_args()


def preprocess(kernel, layer_nb):
    absolute = np.abs(kernel)
    if layer_nb == 1:
        return absolute
    else:
        return np.mean(absolute, 2)


def display_layer(layer_nb, layer):
    nb_kernel = np.shape(layer)[3]
    kernel_size = np.shape(layer)[0]
    if layer_nb == 1:
        depth = np.shape(layer)[2]

    vert_part = []
    k = 0
    for i in range(nb_kernel/16):
        horz_part = []
        for j in range(16):
            horz_part.append(preprocess(layer[:, :, :, k], layer_nb))
            if layer_nb == 1:
                hor_separation = np.ones((kernel_size, 1, depth))
            # else:
            #     hor_separation = np.ones((kernel_size, 1))
                horz_part.append(hor_separation)
            k += 1

        vert_part.append(np.concatenate(horz_part, 1))
        if layer_nb == 1:
            ver_separations = np.ones((1, np.shape(vert_part[0])[1], depth))
            vert_part.append(ver_separations)
        # else:
        #     ver_separations = np.ones((1, np.shape(vert_part[0])[1]))


    layer_image = np.concatenate(vert_part, 0)
    plt.title("Layer " + str(layer_nb))
    plt.axis('off')
    if layer_nb==1:
        plt.imshow(layer_image)
    else:
        plt.imshow(layer_image, cmap='gray')
    plt.show()


for fname in sorted(glob.iglob(args.folder + "*.npy")):
    layer = np.load(fname)
    layer_nb = int(re.search('kernel_layer(.+?).npy', fname).group(1))
    print "Layer nb : " + str(layer_nb)
    display_layer(layer_nb, layer)
