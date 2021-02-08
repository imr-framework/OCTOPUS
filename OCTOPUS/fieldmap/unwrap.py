# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Prepares field map data (from a Siemens scanner) for phase unwrapping with FSL
\nAuthor: Marina Manso Jimeno
\nLast updated: 07/08/2020
'''
import os
import scipy.io as sio
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pydicom import dcmread

from OCTOPUS.utils.dataio import get_data_from_file, read_dicom
from OCTOPUS.recon.rawdata_recon import mask_by_threshold

def fsl_prep(data_path_raw, data_path_dicom, dst_folder, dTE):
    '''
    Prepares a Siemens fieldmap for phase unwrapping using FSL. Saves a phase difference image and a magnitude image with
    the ROI extracted in niftii format to use as inputs for FSL.

    Parameters
    ----------
    data_path_raw : str
        Path to the MATLAB/npy file containing the field map raw data with dimensions [lines, columns, echoes, channels]
    data_path_dicom : str
        Path to the DICOM file containing the phase difference image (.IMA)
    dst_folder : str
        Path to the destination folder where the niftii files are saved
    dTE : float
        Difference in TE between the two echoes in seconds
    '''
    b0_map = get_data_from_file(data_path_raw)
    '''if data_path_raw[-3:] == 'mat':
        b0_map = sio.loadmat(data_path_raw)['b0_map'] # field map raw data
    elif data_path_raw[-3:] == 'npy':
        b0_map = np.load(data_path_raw)
    else:
        raise ValueError('File format not supported, please input a .mat or .npy file')'''


    ##
    # Acq parameters
    ##
    N = b0_map.shape[1] # Matrix Size
    Nchannels = b0_map.shape[-1]

    if len(b0_map.shape) < 5:
        Nslices = 1
        b0_map = b0_map.reshape(b0_map.shape[0], N, 1, 2, Nchannels)
    else:

        Nslices = b0_map.shape[-3]
        sl_order = np.zeros((Nslices))
        sl_order[range(0,Nslices,2)] = range(int(Nslices/2), Nslices)
        sl_order[range(1,Nslices,2)] = range(0, int(Nslices/2))
        sl_order = sl_order.astype(int)

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

    echo1 = echo1[:,:,sl_order,:]
    # Magnitude image with object 'brain' extracted
    mag_im = np.sqrt(np.sum(np.abs(echo1) ** 2, -1))
    mag_im = np.divide(mag_im, np.max(mag_im))


    '''im = np.zeros((128, 128 * 20))
    for i in range(20):
        im[:, int(128 * i):int(128 * (i + 1))] = mag_im[:, :, i]
    plt.imshow(im)
    plt.show()'''



    brain_mask = mask_by_threshold(mag_im)
    brain_extracted = np.squeeze(mag_im) * brain_mask

    # for i in range(Nslices):
    #     plt.imshow(brain_extracted[:,:,i])
    #     plt.show()

    img = nib.Nifti1Image(brain_extracted, np.eye(4))
    nib.save(img, os.path.join(dst_folder, 'mag_vol_extracted.nii.gz'))

    # Phase difference image from DICOM data

    if os.path.isdir(data_path_dicom) or data_path_dicom[-4:] == '.dcm' or data_path_dicom[-4:] == '.IMA':
        vol = read_dicom(data_path_dicom)
        vol = np.rot90(vol[...,sl_order], -1)
    else:
        vol = get_data_from_file(data_path_dicom)
    # Save as niftii

    # for i in range(Nslices):
    #     plt.imshow(vol[:,:,i])
    #     plt.show()

    for i in range(Nslices):
        plt.subplot(1,2,1)
        plt.imshow(brain_extracted[...,i])
        plt.subplot(1,2,2)
        plt.imshow(vol[:,:,i])
        plt.show()
    img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, os.path.join(dst_folder, 'phase_diff.nii.gz'))
