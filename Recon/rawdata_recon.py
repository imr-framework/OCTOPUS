'''
Author: Marina Manso Jimeno
Last modified: 02/28/2020
'''
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import nibabel as nib
import math

from skimage.restoration import unwrap_phase
from pynufft import NUFFT_cpu

def mask_by_threshold(im):
    im = np.squeeze(im)
    mask = np.ones(im.shape)
    threshold = 0.1 * (np.percentile(im, 98) - np.percentile(im, 2)) + np.percentile(im, 2)
    mask[np.where(im < threshold)] = 0
    '''for sl in range(im.shape[-2]):
        for ch in range(im.shape[-1]):
            im_sl = im[:,:,sl, ch]
            mask_sl = mask[:,:, sl, ch]
            threshold = 0.1 * (np.percentile(im_sl, 98) - np.percentile(im_sl, 2)) + np.percentile(
            im_sl, 2)
            mask_sl[np.where(im_sl < threshold)] = 0
            mask[:,:,sl, ch] = mask_sl'''
    return mask

def hermitian_product(echo1, echo2, dTE):
    delta_theta = np.angle(np.sum(echo2 * np.conjugate(echo1), axis=-1))
    '''sos = np.sum(np.abs(echo1), 3)
    sos = np.divide(sos, np.max(sos))
    mask = mask_by_threshold(sos)
    unwrap_deltatheta = unwrap_phase(delta_theta * mask)'''
    b0map = - delta_theta / (dTE * 2 * math.pi)

    return b0map

def separate_channels(echo1, echo2, dTE):
    echo1_ph = np.angle(echo1)
    echo2_ph = np.angle(echo2)

    delta_theta = echo2_ph - echo1_ph
    b0map = - delta_theta / (2 * math.pi * dTE)
    b0map_chcomb = np.zeros(b0map.shape[:-1])
    for sl in range(b0map.shape[-2]):
        for i in range(b0map.shape[0]):
            for j in range(b0map.shape[1]):
                voxel_vals = b0map[i,j,sl,:]
                lowest_quartile = np.quantile(voxel_vals, 0.25)
                highest_quartile = np.quantile(voxel_vals, 0.75)
                trimmed_vals = [val for val in voxel_vals if val > lowest_quartile and val < highest_quartile]
                b0map_chcomb[i,j,sl] = np.mean(np.asarray(trimmed_vals))


    return  b0map_chcomb
def b0map_recon(data_path, method = 'HP', save = 0, plot = 0):
    ##
    # Load the raw data
    ##
    b0_map =  sio.loadmat(data_path + 'rawdata_b0map')['b0_map']

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
            echo1[:, :, sl, ch] = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 0, ch]))
            echo2[:, :, sl, ch] = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 1, ch]))

    # Crop the lines from oversampling factor of 2
    oversamp_factor = int(b0_map.shape[0] / 4)
    echo1 = echo1[oversamp_factor:-oversamp_factor, :, :, :]
    echo2 = echo2[oversamp_factor:-oversamp_factor, :, :, :]

    mag_im = np.sum(np.abs(echo1), 3)
    mag_im = np.divide(mag_im, np.max(mag_im))
    brain_mask = mask_by_threshold(mag_im)
    brain_extracted = np.squeeze(mag_im) * brain_mask
    img = nib.Nifti1Image(brain_extracted, np.eye(4))
    nib.save(img, data_path +'mag_vol_extracted.nii.gz')
    #method = 'SC'
    if method == 'HP':
        b0map = hermitian_product(echo1, echo2, dTE)
    elif method == 'SC':
        b0map = separate_channels(echo1, echo2, dTE)


    # delta_phi_unwrap = unwrap_phase(delta_theta * mask)


    b0map = b0map #* mask
    ##
    # Calculate the field map
    ##
    '''oversamp_factor = int(b0_map.shape[0] / 4)

    b0map_im = np.zeros((N, N, Nslices, Nchannels))
    for ch in range(Nchannels):
        for sl in range(Nslices):
            echo1 = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 0, ch]))
            echo2 = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 1, ch]))
            # Crop the lines from oversampling factor of 2 and calculate phase difference
            echo1 = echo1[oversamp_factor:-oversamp_factor,:]
            echo2 = echo2[oversamp_factor:-oversamp_factor,:]
            # Mask the magnitude images
            echo1 = echo1 * mask_by_threshold(np.abs(echo1))
            echo2 = echo2 * mask_by_threshold(np.abs(echo2))


            delta_theta = np.angle(echo1 / echo2)
            #delta_phi = np.angle(echo1 / echo2)
            #delta_phi = np.arctan((echo1.real * echo2.imag - echo1.imag * echo2.real) / (echo1.real * echo2.real + echo1.imag * echo2.imag))



            #delta_phi_unwrap = unwrap_phase(delta_phi)
            b0map_im[:, :, sl, ch] = delta_theta/ (2 * math.pi * dTE)

    freq_range = [np.min(b0map_im), np.max(b0map_im)]
    sos = np.sum(np.abs(b0map_im), 3)
    sos = np.divide(sos, np.max(sos))
    b0map = np.zeros(sos.shape)
    cv2.normalize(sos, b0map, freq_range[0], freq_range[1], cv2.NORM_MINMAX)
    #b0map = np.expand_dims(b0map_im[:,:,0,-1], axis=2)
    
    #b0map = delta_phi_unwrap / (2* math.pi *dTE)'''
    b0map = np.fliplr(b0map)
    if save:
        np.save(data_path+'b0map', b0map)

    if plot:
        plt.imshow(np.rot90(b0map[:, :, 0], -1), cmap='gray')
        #plt.imshow(np.rot90(np.fliplr(b0map[:,:,0]), -1), cmap='gray') # rotate only for display purposes
        plt.axis('off')
        plt.title('Field Map')
        plt.colorbar()
        plt.show()

    return b0map

def spiral_recon(data_path, ktraj, N, plot = 0):


    ##
    # Load the raw data
    ##
    dat = sio.loadmat(data_path + 'rawdata_spiral')['dat']

    ##
    # Acq parameters
    ##
    Npoints = ktraj.shape[0]
    Nshots = ktraj.shape[1]
    Nchannels = dat.shape[-1]

    if len(dat.shape) < 4:
        Nslices = 1
        dat = dat.reshape(Npoints, Nshots, 1, Nchannels)
    else:
        Nslices = dat.shape[-2]


    if dat.shape[0] != ktraj.shape[0] or dat.shape[1] != ktraj.shape[1]:
        raise ValueError('Raw data and k-space trajectory do not match!')

    ##
    # Arrange data for pyNUFFT
    ##

    om = np.zeros((Npoints * Nshots, 2))
    om[:, 0] = np.real(ktraj).flatten()
    om[:, 1] = np.imag(ktraj).flatten()

    NufftObj = NUFFT_cpu()  # Create a pynufft object
    Nd = (N, N)  # image size
    Kd = (2 * N, 2 * N)  # k-space size
    Jd = (6, 6)  # interpolation size
    NufftObj.plan(om, Nd, Kd, Jd)

    ##
    # Recon
    ##
    im = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
    for ch in range(Nchannels):
        for sl in range(Nslices):
            im[:,:,sl,ch] = NufftObj.solve(dat[:,:,sl,ch].flatten(), solver='cg', maxiter=50)

    sos = np.sum(np.abs(im), 2)
    sos = np.divide(sos, np.max(sos))

    if plot:
        plt.imshow(np.rot90(np.abs(sos[:,:,0]),-1), cmap='gray')
        plt.axis('off')
        plt.title('Uncorrected Image')
        plt.show()
    return