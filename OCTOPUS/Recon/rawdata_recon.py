# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Methods to reconstruct a field map and spiral images from raw data.
\nAuthor: Marina Manso Jimeno
\nLast modified: 07/16/2020
'''
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import nibabel as nib
import math

from pynufft import NUFFT_cpu

def mask_by_threshold(im):
    '''
    Masks a magnitude image by thresholding according to:
    Jenkinson M. (2003). Fast, automated, N-dimensional phase-unwrapping algorithm. Magnetic resonance in medicine, 49(1), 193–197. https://doi.org/10.1002/mrm.10354

    Parameters
    ----------
    im : np.ndarray
        Magnitude image

    Returns
    -------
    mask : np.ndarray
        Binary image for background segmentation
    '''
    im = np.squeeze(im)
    mask = np.ones(im.shape)
    threshold = 0.1 * (np.percentile(im, 98) - np.percentile(im, 2)) + np.percentile(im, 2)
    mask[np.where(im < threshold)] = 0
    return mask

def hermitian_product(echo1, echo2, dTE):
    '''
    Calculates the phase difference and frequency map given two echo data and deltaTE between them.
    Channels are combined using the hermitian product as described by:
    Robinson, S., & Jovicich, J. (2011). B0 mapping with multi-channel RF coils at high field. Magnetic resonance in medicine, 66(4), 976–988. https://doi.org/10.1002/mrm.22879

    Parameters
    ----------
    echo1 : np.ndarray
        Complex image corresponding to the first echo with dimensions [N, N, Nslices, Nchannels]
    echo2 : np.ndarray
        Complex image corresponding to the second echo with same dimensions as echo1
    dTE : float
        TE difference between the two echos in seconds.

    Returns
    -------
    fmap : np.ndarray
        Frequency map in Hz with dimensions [N, N, Nslices]
    '''

    delta_theta = np.angle(np.sum(echo2 * np.conjugate(echo1), axis=-1))
    fmap = - delta_theta / (dTE * 2 * math.pi)

    return fmap

def separate_channels(echo1, echo2, dTE):
    '''
    Calculates the phase difference and frequency map given two echo data and deltaTE between them.
    Channels are combined using a trimmed average method inspired by the separate channels method described by:
    Robinson, S., & Jovicich, J. (2011). B0 mapping with multi-channel RF coils at high field. Magnetic resonance in medicine, 66(4), 976–988. https://doi.org/10.1002/mrm.22879

    Parameters
    ----------
    echo1 : np.ndarray
        Complex image corresponding to the first echo with dimensions [N, N, Nslices, Nchannels]
    echo2 : np.ndarray
        Complex image corresponding to the second echo with same dimensions as echo1
    dTE : float
        TE difference between the two echos in seconds.

    Returns
    -------
    fmap_chcomb : np.ndarray
        Frequency map in Hz with dimensions [N, N, Nslices]
    '''
    echo1_ph = np.angle(echo1)
    echo2_ph = np.angle(echo2)

    delta_theta = echo2_ph - echo1_ph
    fmap = - delta_theta / (2 * math.pi * dTE)
    fmap_chcomb = np.zeros(fmap.shape[:-1])
    for sl in range(fmap.shape[-2]):
        for i in range(fmap.shape[0]):
            for j in range(fmap.shape[1]):
                voxel_vals = fmap[i,j,sl,:]
                lowest_quartile = np.quantile(voxel_vals, 0.25)
                highest_quartile = np.quantile(voxel_vals, 0.75)
                trimmed_vals = [val for val in voxel_vals if val > lowest_quartile and val < highest_quartile]
                fmap_chcomb[i,j,sl] = np.mean(np.asarray(trimmed_vals))

    return fmap_chcomb

def fmap_recon(data_path, method = 'HP', save = 0, plot = 0):
    '''
    Frequency map reconstruction from dual echo raw data

    Parameters
    ----------
    data_path : str
        Path containing the raw data .mat file
    method : str
        Method for channel combination. Options are 'HP' or 'SC'. Default is 'HP'.
    save : bool
        Saving the data in a .npy file option. Default is 0 (not save).
    plot : bool
        Plotting a slice of the reconstructed frequency map option. Default is 0 (not plot).

    Returns
    -------
    fmap : np.ndarray
        Frequency map in Hz
    '''
    ##
    # Load the raw data
    ##
    f_map =  sio.loadmat(data_path)['b0_map']

    ##
    # Acq parameters
    ##
    dTE = 2.46e-3 # seconds
    N = f_map.shape[1] # Matrix Size
    Nchannels = f_map.shape[-1]

    if len(f_map.shape) < 5:
        Nslices = 1
        f_map = f_map.reshape(f_map.shape[0], N, 1, 2, Nchannels)
    else:

        Nslices = f_map.shape[2]

    ##
    # FT to get the echo complex images
    ##

    echo1 = np.zeros((N * 2, N, Nslices, Nchannels), dtype=complex)
    echo2 = np.zeros((N * 2, N, Nslices, Nchannels), dtype=complex)
    for ch in range(Nchannels):
        for sl in range(Nslices):
            echo1[:, :, sl, ch] = fft.fftshift(fft.ifft2(f_map[:, :, sl, 0, ch]))
            echo2[:, :, sl, ch] = fft.fftshift(fft.ifft2(f_map[:, :, sl, 1, ch]))

    # Crop the lines from oversampling factor of 2
    oversamp_factor = int(f_map.shape[0] / 4)
    echo1 = echo1[oversamp_factor:-oversamp_factor, :, :, :]
    echo2 = echo2[oversamp_factor:-oversamp_factor, :, :, :]

    ##
    # Calculate the field map
    ##
    if method == 'HP':
        fmap = hermitian_product(echo1, echo2, dTE)
    elif method == 'SC':
        fmap = separate_channels(echo1, echo2, dTE)
    else:
        raise ValueError('The method you specified is not supported')

    fmap = np.fliplr(fmap)
    if save:
        np.save(data_path+'fmap', fmap)

    if plot:
        mid = math.floor(fmap.shape[-1]/2)
        plt.imshow(np.rot90(fmap[:, :, mid], -1), cmap='gray')
        #plt.imshow(np.rot90(np.fliplr(b0map[:,:,0]), -1), cmap='gray') # rotate only for display purposes
        plt.axis('off')
        plt.title('Field Map')
        plt.colorbar()
        plt.show()

    return fmap

def spiral_recon(data_path, dst_folder, ktraj, N, plot = 0):
    '''
    Spiral image reconstruction from raw data

    Parameters
    ----------
    data_path : str
        Path containing the raw data .mat file
    dst_folder : str
        Path to the folder where the reconstructed image is saved
    ktraj : np.ndarray
        k-space trajectory coordinates with dimensions [Npoints, Nshots]
    N : int
        Matrix size of the reconstructed image
    plot : bool
        Plotting a slice of the reconstructed image option. Default is 0 (not plot).
    '''

    ##
    # Load the raw data
    ##
    dat = sio.loadmat(data_path)['dat']
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
    ktraj_sc = math.pi / abs(np.max(ktraj))
    ktraj = ktraj * ktraj_sc  # pyNUFFT scaling [-pi, pi]
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

    sos = np.sum(np.abs(im), -1)
    sos = np.divide(sos, np.max(sos))
    np.save(dst_folder + 'uncorrected_spiral.npy', sos)

    if plot:
        plt.imshow(np.rot90(np.abs(sos[:,:,0]),-1), cmap='gray')
        plt.axis('off')
        plt.title('Uncorrected Image')
        #plt.savefig('foo.png')
        plt.show()

    return sos