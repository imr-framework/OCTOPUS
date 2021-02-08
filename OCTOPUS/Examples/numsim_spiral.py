# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Numerical Simulation Experiments with spiral trajectory
Author: Marina Manso Jimeno
Last updated: 07/15/2020
'''
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.segmentation import flood_fill

import OCTOPUS.ORC as ORC
import OCTOPUS.fieldmap.simulate as fieldmap_sim
from OCTOPUS.utils.plotting import plot_correction_results
from OCTOPUS.utils.metrics import create_table
from OCTOPUS.recon.rawdata_recon import mask_by_threshold
##
# Original image: Shep-Logan Phantom
##
def numsim_spiral():
    N = 192  # ph.shape[0]
    ph = resize(shepp_logan_phantom(), (N, N)).astype(complex)
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
    # Spiral k-space trajectory
    ##
    dt = 10e-6
    # ktraj = np.load('sample_data/ktraj_sutton2003.npy')
    ktraj = np.load('sample_data/ktraj.npy') # k-space trajectory

    plt.plot(ktraj.real,ktraj.imag)
    plt.title('Spiral trajectory')
    plt.show()

    #ktraj_dcf = np.load('test_data/ktraj_noncart_dcf.npy').flatten() # density compensation factor
    t_ro = ktraj.shape[0] * dt
    T = (np.arange(ktraj.shape[0]) * dt).reshape(ktraj.shape[0],1)

    seq_params = {'N': ph.shape[0], 'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 't_readout': t_ro, 't_vector': T}#, 'dcf': ktraj_dcf}
    ##
    # Simulated field map
    ##
    fmax_v = [250, 500, 750] # Hz

    i = 0
    or_corrupted = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_CPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_fsCPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_MFI = np.zeros((N, N, len(fmax_v)), dtype='complex')
    for fmax  in fmax_v:

        field_map = fieldmap_sim.realistic(np.abs(ph), mask, fmax)
        ### For reproducibility
        # dst = np.zeros((N, N))
        # field_map = cv2.normalize(np.load('M2.npy'), dst, -fmax, fmax, cv2.NORM_MINMAX)
        # field_map = field_map * np.load('mask.npy')
        # field_map = fieldmap_gen.fieldmap_bin(field_map,5)
        ###
        plt.imshow(field_map, cmap='gray')
        plt.title('Field Map +/-' + str(fmax) + ' Hz')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    ##
    # Corrupted images
    ##
        or_corrupted[:,:,i], _ = ORC.add_or_CPR(ph, ktraj, field_map, 1, seq_params)
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
        or_corrected_CPR[:, :, i] = ORC.CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 1, seq_params)

        or_corrected_fsCPR[:, :, i] = ORC.fs_CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 2, 1, seq_params)

        or_corrected_MFI[:,:,i] = ORC.MFI(or_corrupted[:,:,i], 'im', ktraj, field_map, 2, 1, seq_params)

        i += 1

    ##
    # Plot
    ##
    im_stack = np.stack((np.squeeze(or_corrupted), np.squeeze(or_corrected_CPR), np.squeeze(or_corrected_fsCPR), np.squeeze(or_corrected_MFI)))
    #np.save('im_stack.npy',im_stack)
    cols = ('Corrupted Image','CPR Correction', 'fs-CPR Correction', 'MFI Correction')
    row_names = ('-/+ 250 Hz', '-/+ 500 Hz', '-/+ 750 Hz')
    plot_correction_results(im_stack, cols, row_names)


    ##
    # Metrics
    ##
    #im_stack = np.stack((np.dstack((ph, ph, ph)), or_corrupted, or_corrected_CPR, or_corrected_fsCPR, or_corrected_MFI))
    #create_table(im_stack, cols, row_names)

if __name__ == "__main__":
    numsim_spiral()