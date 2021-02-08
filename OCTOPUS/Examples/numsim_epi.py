# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Numerical Simulation Experiments with EPI trajectory
Author: Marina Manso Jimeno
Last updated: 01/19/2021
'''
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.segmentation import flood_fill

import OCTOPUS.ORC as ORC
import OCTOPUS.fieldmap.simulate as fieldmap_sim
from OCTOPUS.recon.rawdata_recon import mask_by_threshold
from OCTOPUS.utils.plotting import plot_correction_results


def ssEPI_2d(N, FOV):
    '''
    Generates a single-shot EPI trajectory

    Parameters
    ----------
    N : int
        Matrix size
    FOV : float
        FOV in meters

    Returns
    -------
    ktraj : numpy.ndarray
        k-space trajcectory in 1/m
    '''
    kmax2 = N / FOV
    kx = np.arange(int(-kmax2 / 2), int(kmax2 / 2) + 1, kmax2 / (N - 1))
    ky = np.arange(int(-kmax2 / 2), int(kmax2 / 2) + 1, kmax2 / (N - 1))
    ktraj = []  # np.zeros((N**2, 2))
    for i, ky_i in enumerate(ky):
        for kx_i in kx:
            if i % 2 == 0:
                ktraj.append([kx_i, ky_i])
            else:
                ktraj.append([-kx_i, ky_i])
    ktraj = np.stack(ktraj)
    return ktraj

def numsim_epi():
    ##
    # Original image: Shep-Logan Phantom
    ##
    N = 192
    FOV = 384e-3
    ph = resize(shepp_logan_phantom(), (N,N)).astype(complex)
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
    # EPI k-space trajectory
    ##
    dt = 10e-6
    ktraj = ssEPI_2d(N, FOV)  # k-space trajectory

    plt.plot(ktraj[:,0], ktraj[:,1])
    plt.title('EPI trajectory')
    plt.show()

    Ta = ktraj.shape[0] * dt
    T = (np.arange(ktraj.shape[0]) * dt).reshape(ktraj.shape[0], 1)
    seq_params = {'N': N, 'Npoints': ktraj.shape[0], 'Nshots': 1, 't_readout': Ta,
                      't_vector': T}

    ##
    # Simulated field map
    ##
    fmax_v = [50, 75, 100]  # Hz
    
    i = 0
    or_corrupted = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_CPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_fsCPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
    or_corrected_MFI = np.zeros((N, N, len(fmax_v)), dtype='complex')
    for fmax in fmax_v:
        field_map = fieldmap_sim.realistic(np.abs(ph), mask, fmax)
        
        plt.imshow(field_map, cmap='gray')
        plt.title('Field Map +/-' + str(fmax) + ' Hz')
        plt.colorbar()
        plt.axis('off')
        plt.show()

        ##
        # Corrupted images
        ##
        or_corrupted[:, :, i], _ = ORC.add_or_CPR(ph, ktraj, field_map, 'EPI', seq_params)
        corrupt = (np.abs(or_corrupted[:, :, i]) - np.abs(or_corrupted[..., i]).min()) / (
                    np.abs(or_corrupted[:, :, i]).max() - np.abs(or_corrupted[..., i]).min())
        # plt.imshow(np.abs(or_corrupted[:,:,i]),cmap='gray')
        plt.imshow(corrupt, cmap='gray')
        plt.colorbar()
        plt.title('Corrupted Image')
        plt.axis('off')
        plt.show()

        ###
        # Corrected images
        ###
        or_corrected_CPR[:, :, i] = ORC.CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 'EPI', seq_params)

        or_corrected_fsCPR[:, :, i] = ORC.fs_CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 2, 'EPI', seq_params)

        or_corrected_MFI[:, :, i] = ORC.MFI(or_corrupted[:, :, i], 'im', ktraj, field_map, 2, 'EPI', seq_params)

        i += 1

    ##
    # Plot
    ##
    im_stack = np.stack((np.squeeze(or_corrupted), np.squeeze(or_corrected_CPR), np.squeeze(or_corrected_fsCPR),
                         np.squeeze(or_corrected_MFI)))
    # np.save('im_stack.npy',im_stack)
    cols = ('Corrupted Image', 'CPR Correction', 'fs-CPR Correction', 'MFI Correction')
    row_names = ('-/+ 250 Hz', '-/+ 500 Hz', '-/+ 750 Hz')
    plot_correction_results(im_stack, cols, row_names)
if __name__ == "__main__":
    numsim_epi()