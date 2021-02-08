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
from OCTOPUS.recon.rawdata_recon import mask_by_threshold

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.util import img_as_float, random_noise
from skimage.segmentation import flood_fill
##
# Original image: Shep-Logan Phantom
##

def numsim_cartesian():

    N = 192
    ph = resize(shepp_logan_phantom(), (N,N))


    ph = ph.astype(complex)
    plt.imshow(np.abs(ph), cmap='gray')
    plt.title('Original phantom')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    brain_mask = mask_by_threshold(ph)

    # Floodfill from point (0, 0)
    ph_holes = ~(flood_fill(brain_mask, (0, 0), 1).astype(int)) + 2
    mask = brain_mask + ph_holes

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


    or_corrupted = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_CPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_fsCPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_MFI = np.zeros((N, N, len(fmax_v)), dtype='complex')
    for fmax in fmax_v:
        field_map = fieldmap_sim.realistic(np.abs(ph), mask, fmax)


        plt.imshow(field_map, cmap='gray')
        plt.title('Field Map')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    ##
    # Corrupted images
    ##
        or_corrupted[:,:,i],_ = ORC.add_or_CPR(ph, ktraj_cart, field_map)
        corrupt = (np.abs(or_corrupted[:,:,i]) - np.abs(or_corrupted[...,i]).min())/(np.abs(or_corrupted[:,:,i]).max() - np.abs(or_corrupted[...,i]).min())
        #plt.imshow(np.abs(or_corrupted[:,:,i]),cmap='gray')
        plt.imshow(corrupt, cmap='gray')
        plt.colorbar()
        plt.title('Corrupted Image')
        plt.axis('off')
        plt.show()

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

    # np.save('or_corrupted.npy', or_corrupted)
    # np.save('or_corrected_CPR.npy', or_corrected_CPR)
    # np.save('or_corrected_fsCPR.npy', or_corrected_fsCPR)
    # np.save('or_corrected_MFI.npy', or_corrected_MFI)
    ##
    # Metrics
    ##
    #im_stack = np.stack((np.dstack((ph, ph, ph)), or_corrupted, or_corrected_CPR, or_corrected_fsCPR, or_corrected_MFI))
    #create_table(im_stack, cols, row_names)

if __name__ == "__main__":
    numsim_cartesian()