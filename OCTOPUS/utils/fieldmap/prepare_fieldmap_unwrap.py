# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Prepares field map data (from a Siemens scanner) for phase unwrapping with FSL
Author: Marina Manso Jimeno
Last updated: 07/08/2020
'''
import os
import scipy.io as sio
import numpy as np
import nibabel as nib

from pydicom import dcmread

from OCTOPUS.Recon.rawdata_recon import mask_by_threshold

data_path = os.path.abspath('../../Data/20200708/shim/')

b0_map = sio.loadmat(os.path.join(data_path, 'rawdata_b0map'))['b0_map'] # field map raw data

##
# Acq parameters
##
dTE = 2.46e-3 # seconds
N = b0_map.shape[1] # Matrix Size
Nchannels = b0_map.shape[-1]

if len(b0_map.shape) < 5:
    Nslices = 1
    b0_map = b0_map.reshape(b0_map.shape[0], N, 1, 2, Nchannels)
else:

    Nslices = b0_map.shape[-2]

##
# FT to get the echo complex images
##

echo1 = np.zeros((N * 2, N, Nslices, Nchannels), dtype=complex)
echo2 = np.zeros((N * 2, N, Nslices, Nchannels), dtype=complex)
for ch in range(Nchannels):
    for sl in range(Nslices):
        echo1[:, :, sl, ch] = np.fft.fftshift(np.fft.ifft2(b0_map[:, :, sl, 0, ch]))
        echo2[:, :, sl, ch] = np.fft.fftshift(np.fft.ifft2(b0_map[:, :, sl, 1, ch]))

# Crop the lines from oversampling factor of 2
oversamp_factor = int(b0_map.shape[0] / 4)
echo1 = echo1[oversamp_factor:-oversamp_factor, :, :, :]
echo2 = echo2[oversamp_factor:-oversamp_factor, :, :, :]

# Magnitude image with object 'brain' extracted
mag_im = np.sum(np.abs(echo1), 3)
mag_im = np.divide(mag_im, np.max(mag_im))
brain_mask = mask_by_threshold(mag_im)
brain_extracted = np.squeeze(mag_im) * brain_mask
img = nib.Nifti1Image(brain_extracted, np.eye(4))
nib.save(img, os.path.join(data_path, 'mag_vol_extracted.nii.gz'))

# Phase difference image from DICOM data
data = dcmread(os.path.join(data_path, 'phase_diff_map_dicom.IMA'))
# Get the image data
vol = data.pixel_array
# Save as niftii
img = nib.Nifti1Image(np.rot90(vol), np.eye(4))
nib.save(img, os.path.join(data_path, 'phase_diff.nii.gz'))
