# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Calculates entropy for the correction images as well as for the gold standard
Author: Marina Manso Jimeno
Last updated: 07/08/2020
'''
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from OCTOPUS.Recon.rawdata_recon import mask_by_threshold

def entropy_im(I):
    I_norm = np.zeros(I.shape)
    I_norm = cv2.normalize(I, I_norm, 0, 255, cv2.NORM_MINMAX).astype(int)

    p = np.histogram(I_norm, 255)[0]
    p = p[p != 0]
    p = p / I.size

    entropy = - np.sum([p_i * math.log2(p_i) for p_i in p])
    return entropy


data_path = '../../Data/20200708/no_shim/'
images = ['uncorrected_spiral.npy', 'corrections/CPR.npy', 'corrections/fsCPR_Lx2.npy', 'corrections/MFI_Lx2.npy']


entropy = []
for image in images:
    im = np.load(data_path + image)

    if len(im.shape) > 3:
        sos = np.divide(np.sum(np.abs(im), -1), np.max(np.sum(np.abs(im), -1)))
    else:
        sos = im
    mask = mask_by_threshold(sos)
    plt.imshow(sos[:, :, 0] * mask)
    plt.show()
    entropy.append(entropy_im(np.squeeze(sos) * mask))

print(entropy)


