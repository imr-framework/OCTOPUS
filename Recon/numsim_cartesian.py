'''
Numerical Simulation Experiments with Cartesian trajectories
Author: Marina Manso Jimeno
Last updated: 05/20/2020
'''
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import Recon.fieldmap_gen as fieldmap_gen
import Recon.ORC as ORC
from utils.plot_restults import plot_correction_results

##
# Original image: Shep-Logan Phantom
##
ph = np.load('test_data/slph_im.npy').astype(complex) # Shep-Logan Phantom
#ph = sio.loadmat('test_data/sl_ph.mat')['sl_ph']
ph = (ph - np.min(ph)) / (np.max(ph)-np.min(ph)) # Normalization
N = ph.shape[0]
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
plt.show()

##
# Simulated field map
##
fmax_v = [2500, 5000, 7500] # Hz

i = 0
or_corrupted = np.zeros((N, N, len(fmax_v)), dtype='complex')
or_corrected_CPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
or_corrected_fsCPR = np.zeros((N, N, len(fmax_v)), dtype='complex')
or_corrected_MFI = np.zeros((N, N, len(fmax_v)), dtype='complex')
for fmax in fmax_v:
    #field_map = fieldmap_gen.spherical_order4(N,fmax)
    #field_map = fieldmap_gen.parabolic(N, fmax)
    field_map = fieldmap_gen.hyperbolic(N, fmax)
    #field_map = fieldmap_gen.realistic(np.abs(ph), fmax)
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
row_names = ('-/+ 2500 Hz', '-/+ 5000 Hz', '-/+ 7500 Hz')
plot_correction_results(im_stack, cols, row_names)
# Corrupted images
im2plot_corrupted = np.vstack((np.hstack((ph, or_corrupted[:,:,0])), np.hstack((or_corrupted[:,:,1], or_corrupted[:,:,2]))))
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
plt.show()

# Difference images
im2plot_diff1 = np.vstack((np.hstack((ph - ph, or_corrected_CPR[:,:,0] - ph)), np.hstack((or_corrected_fsCPR[:,:,0] - ph, or_corrected_MFI[:,:,0] - ph))))
plt.imshow(np.abs(im2plot_diff1),cmap='gray')
plt.axis('off')
plt.title('Difference Image 2500 Hz')
plt.show()

im2plot_diff2 = np.vstack((np.hstack((ph - ph, or_corrected_CPR[:,:,1] - ph)), np.hstack((or_corrected_fsCPR[:,:,1] - ph, or_corrected_MFI[:,:,1] - ph))))
plt.imshow(np.abs(im2plot_diff2),cmap='gray')
plt.axis('off')
plt.title('Difference Image 5000 Hz')
plt.show()

im2plot_diff3 = np.vstack((np.hstack((ph - ph, or_corrected_CPR[:,:,2] - ph)), np.hstack((or_corrected_fsCPR[:,:,2] - ph, or_corrected_MFI[:,:,2] - ph))))
plt.imshow(np.abs(im2plot_diff3),cmap='gray')
plt.axis('off')
plt.title('Difference Image 7500 Hz')
plt.show()

im2plot_diff_2500 = np.hstack((or_corrected_CPR[:,:,0] - or_corrected_fsCPR[:,:,0], or_corrected_CPR[:,:,0]- or_corrected_MFI[:,:,0], or_corrected_fsCPR[:,:,0] - or_corrected_MFI[:,:,0]))
plt.imshow(np.abs(im2plot_diff_2500),cmap='gray')
plt.axis('off')
plt.title('Correction difference at 2500 Hz')
plt.show()

im2plot_diff_5000 = np.hstack((or_corrected_CPR[:,:,1] - or_corrected_fsCPR[:,:,1], or_corrected_CPR[:,:,1]- or_corrected_MFI[:,:,1], or_corrected_fsCPR[:,:,1] - or_corrected_MFI[:,:,1]))
plt.imshow(np.abs(im2plot_diff_5000),cmap='gray')
plt.axis('off')
plt.title('Correction difference at 5000 Hz')
plt.show()

im2plot_diff_7500 = np.hstack((or_corrected_CPR[:,:,2] - or_corrected_fsCPR[:,:,2], or_corrected_CPR[:,:,2]- or_corrected_MFI[:,:,2], or_corrected_fsCPR[:,:,2] - or_corrected_MFI[:,:,2]))
plt.imshow(np.abs(im2plot_diff_7500),cmap='gray')
plt.axis('off')
plt.title('Correction difference at 7500 Hz')
plt.show()