'''
Numerical Simulation Experiments
Author: Marina Manso Jimeno
Last updated: 05/26/2020
'''
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

import Recon.ORC as ORC
import Recon.fieldmap_gen as fieldmap_gen
from utils.metrics import create_table
from utils.plot_restults import plot_correction_results

##
# Original image: Shep-Logan Phantom
##
ph = np.load('test_data/slph_im.npy').astype(complex) # Shep-Logan Phantom
ph = (ph - np.min(ph)) / (np.max(ph)-np.min(ph)) # Normalization
N = ph.shape[0]
plt.imshow(np.abs(ph), cmap='gray')
plt.title('Original phantom')
plt.axis('off')
plt.colorbar()
plt.show()

##
# Spiral k-space trajectory
##
dt = 10e-6
ktraj = np.load('test_data/ktraj_noncart.npy') # k-space trajectory
ktraj_sc = math.pi / abs(np.max(ktraj))
ktraj = ktraj * ktraj_sc # pyNUFFT scaling [-pi, pi]
plt.plot(ktraj.real,ktraj.imag)
plt.title('Spiral trajectory')
plt.show()

ktraj_dcf = np.load('test_data/ktraj_noncart_dcf.npy').flatten() # density compensation factor
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
    field_map = fieldmap_gen.spherical_order4(N, fmax)
    #field_map = fieldmap_gen.hyperbolic(N, fmax)
    plt.imshow(field_map, cmap='gray')
    plt.title('Field Map +/-' + str(fmax) + ' Hz')
    plt.colorbar()
    plt.axis('off')
    plt.show()

##
# Corrupted images
##
    or_corrupted[:,:,i] = ORC.add_or_CPR(ph, ktraj, field_map, 1, seq_params)
    plt.imshow(np.abs(or_corrupted[:,:,i]),cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.show()

###
# Corrected images
###
    or_corrected_CPR[:, :, i] = ORC.CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 1, seq_params)
    print('CPR done')
    or_corrected_fsCPR[:, :, i] = ORC.fs_CPR(or_corrupted[:, :, i], 'im', ktraj, field_map, 2, 1, seq_params)
    print('fs-CPR done')
    or_corrected_MFI[:,:,i] = ORC.MFI(or_corrupted[:,:,i], 'im', ktraj, field_map, 2, 1, seq_params)
    print('MFI done')
    i += 1

##
# Plot
##

# Metrics
im_stack = np.stack((np.squeeze(or_corrupted), np.squeeze(or_corrected_CPR), np.squeeze(or_corrected_fsCPR), np.squeeze(or_corrected_MFI)))
cols = ('CPR', 'fs-CPR', 'MFI')
row_names = ('-/+ 250 Hz', '-/+ 500 Hz', '-/+ 750 Hz')
plot_correction_results(im_stack, cols, row_names)
create_table(im_stack, cols)

# Corrupted images
'''im2plot_corrupted = np.vstack((np.hstack((ph, or_corrupted[:,:,0])), np.hstack((or_corrupted[:,:,1], or_corrupted[:,:,2]))))
plt.imshow(np.abs(im2plot_corrupted),cmap='gray')
plt.axis('off')
plt.show()

# Corrected images
im2plot_corrected_CPR = np.vstack((np.hstack((ph, or_corrected_CPR[:,:,0])), np.hstack((or_corrected_CPR[:,:,1], or_corrected_CPR[:,:,2]))))
plt.imshow(np.abs(im2plot_corrected_CPR),cmap='gray')
plt.axis('off')
plt.title('CPR correction')
plt.show()

im2plot_corrected_fsCPR = np.vstack((np.hstack((ph, or_corrected_fsCPR[:,:,0])), np.hstack((or_corrected_fsCPR[:,:,1], or_corrected_fsCPR[:,:,2]))))
plt.imshow(np.abs(im2plot_corrected_fsCPR),cmap='gray')
plt.axis('off')
plt.title('fs-CPR correction')
plt.show()

im2plot_corrected_MFI = np.vstack((np.hstack((ph, or_corrected_MFI[:,:,0])), np.hstack((or_corrected_MFI[:,:,1], or_corrected_MFI[:,:,2]))))
plt.imshow(np.abs(im2plot_corrected_MFI),cmap='gray')
plt.axis('off')
plt.title('MFI correction')
plt.show()'''

