'''
Author: Marina Manso Jimeno
Last modified: 02/28/2020
'''
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import cv2
import math

from pynufft import NUFFT_cpu

def b0map_recon(data_path, save = 0, plot = 0):
    ##
    # Load the raw data
    ##
    b0_map = sio.loadmat(data_path + 'acrph_df')['b0_map'] # sio.loadmat(data_path + 'rawdata_b0map')['b0_map']

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
    # Calculate the field map
    ##
    b0map_im = np.zeros((N, N, Nslices, Nchannels))
    for ch in range(Nchannels):
        for sl in range(Nslices):
            echo1 = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 0, ch]))
            echo2 = fft.fftshift(fft.ifft2(b0_map[:, :, sl, 1, ch]))
            # Crop the lines from oversampling factor of 2 and calculate phase difference
            phi = np.angle(echo1[int(echo1.shape[0] / 4):-int(echo1.shape[0] / 4), :] / echo2[int(echo1.shape[0] / 4):-int(echo1.shape[0] / 4), :])
            b0map_im[:, :, sl, ch] = phi / (2 * math.pi * dTE)

    freq_range = [np.min(b0map_im), np.max(b0map_im)]
    sos = np.sum(np.abs(b0map_im), 3)
    sos = np.divide(sos, np.max(sos))
    b0map = np.zeros(sos.shape)
    cv2.normalize(sos, b0map, np.min(b0map_im), np.max(b0map_im), cv2.NORM_MINMAX)
    b0map = np.fliplr(b0map)
    if save:
        np.save(data_path+'b0map', b0map)

    if plot:
        plt.imshow(np.rot90(b0map[:,:,0], -1), cmap='gray') # rotate only for display purposes
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