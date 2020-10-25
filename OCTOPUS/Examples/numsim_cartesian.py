# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Numerical Simulation Experiments with Cartesian trajectories
Author: Marina Manso Jimeno
Last updated: 07/15/2020
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

import OCTOPUS.fieldmap.simulate as fieldmap_sim
import OCTOPUS.ORC as ORC
from OCTOPUS.utils.plotting import plot_correction_results
from OCTOPUS.utils.metrics import create_table

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
##
# Original image: Shep-Logan Phantom
##

def numsim_cartesian():
    '''ph = np.load('sample_data/slph_im.npy').astype(complex) # Shep-Logan Phantom
    ph = (ph - np.min(ph)) / (np.max(ph)-np.min(ph)) # Normalization'''
    N = 192#ph.shape[0]
    ph = resize(shepp_logan_phantom(), (N,N)).astype(complex)
    plt.imshow(np.abs(ph), cmap='gray')
    plt.title('Original phantom')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    ##
    # Cartesian k-space trajectory
    ##
    dt = 10e-6 # grad raster time
    ktraj_cart = np.arange(0, N * dt, dt).reshape(1,N)
    ktraj_cart = np.tile(ktraj_cart, (N, 1))
    plt.imshow(ktraj_cart, cmap='gray')
    plt.title('Cartesian trajectory')
    plt.show()

    ##
    # Simulated field map
    ##
    fmax_v = [1600, 3200, 4800] # Hz correspontig to 25, 50 and 75 ppm at 3T
    i = 0

    field_maps = np.zeros((N, N, len(fmax_v)))
    or_corrupted = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_CPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_fsCPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_MFI = np.zeros((N, N, len(fmax_v)), dtype='complex')
    for fmax in fmax_v:
        field_map = fieldmap_sim.realistic(np.abs(ph), fmax)

        ### For reproducibility
        # dst = np.zeros((N,N))
        # field_map = cv2.normalize(np.load('M2.npy'), dst, -fmax, fmax, cv2.NORM_MINMAX)
        # field_map = field_map * np.load('mask.npy')
        ###

        field_maps[:,:,i] = field_map
        plt.imshow(field_map, cmap='gray')
        plt.title('Field Map')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    ##
    # Corrupted images
    ##
        or_corrupted[:,:,i] = ORC.add_or_CPR(ph, ktraj_cart, field_map)
        '''plt.imshow(np.abs(or_corrupted[:,:,i]),cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.show()'''

    ###
    # Corrected images
    ###
        or_corrected_CPR[:, :, i] = ORC.CPR(or_corrupted[:, :, i], 'im', ktraj_cart, field_map)
        or_corrected_fsCPR[:, :, i] = ORC.fs_CPR(or_corrupted[:, :, i], 'im', ktraj_cart, field_map, 2)
        or_corrected_MFI[:,:,i] = ORC.MFI(or_corrupted[:,:,i], 'im', ktraj_cart, field_map, 2)
        i += 1

##
# Plot
##
    im_stack = np.stack((np.squeeze(or_corrupted), np.squeeze(or_corrected_CPR), np.squeeze(or_corrected_fsCPR), np.squeeze(or_corrected_MFI)))
    cols = ('Corrupted Image', 'CPR Correction', 'fs-CPR Correction', 'MFI Correction')
    row_names = ('-/+ 1600 Hz', '-/+ 3200 Hz', '-/+ 4800 Hz')
    plot_correction_results(im_stack, cols, row_names)

    ##
    # Metrics
    ##
    #im_stack = np.stack((np.dstack((ph, ph, ph)), or_corrupted, or_corrected_CPR, or_corrected_fsCPR, or_corrected_MFI))
    #create_table(im_stack, cols, row_names)

if __name__ == "__main__":
    numsim_cartesian()