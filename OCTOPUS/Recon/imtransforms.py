# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Methods to do k-space to image and image to k-space for both Cartesian and non-cartesian data
\nAuthor: Marina Manso Jimeno
\nLast updated: 07/18/2020
'''

import numpy.fft as npfft
import numpy as np
import pynufft

from math import pi

def im2ksp(M, cartesian_opt, NufftObj=None, params=None):
    '''Image to k-space transformation

    Parameters
    ----------
    M : numpy.ndarray
        Image data
    cartesian_opt : int
        Cartesian = 1, Non-Cartesian = 0.
    NufftObj : pynufft.linalg.nufft_cpu.NUFFT_cpu
        Non-uniform FFT Object for non-cartesian transformation. Default is None.
    params : dict
        Sequence parameters. Default is None.

    Returns
    -------
    kspace : numpy.ndarray
        k-space data
    '''

    if cartesian_opt == 1:

        kspace = npfft.fftshift(npfft.fft2(M))
    elif cartesian_opt == 0:
        # Sample phantom along ktraj
        if 'Npoints' not in params:
            raise ValueError('The number of acquisition points is missing')
        if 'Nshots' not in params:
            raise ValueError('The number of shots is missing')
        kspace = NufftObj.forward(M).reshape((params['Npoints'], params['Nshots']))  # sampled image
    else:
        raise ValueError('Cartesian option should be either 0 or 1')
    return kspace

def ksp2im(ksp, cartesian_opt, NufftObj=None, params=None):
    '''K-space to image transformation

    Parameters
    ----------
    ksp : numpy.ndarray
        K-space data
    cartesian_opt : int
        Cartesian = 1, Non-Cartesian = 0.
    NufftObj : pynufft.linalg.nufft_cpu.NUFFT_cpu
        Non-uniform FFT Object for non-cartesian transformation. Default is None.
    params : dict
        Sequence parameters. Default is None.

    Returns
    -------
    im : numpy.ndarray
        Image data
    '''
    if cartesian_opt == 1:
        im = npfft.ifft2(npfft.fftshift(ksp))

    elif cartesian_opt == 0:
        if 'dcf' in params:
            ksp_dcf = ksp.reshape((params['Npoints']*params['Nshots'],))*params['dcf']
            im = NufftObj.adjoint(ksp_dcf)  # * np.prod(sqrt(4 * params['N'] ** 2))
        else:
            ksp_dcf = ksp.reshape((params['Npoints'] * params['Nshots'],))
            im = NufftObj.solve(ksp_dcf, solver='cg', maxiter=50)



    else:
        raise ValueError('Cartesian option should be either 0 or 1')

    return im

def nufft_init(kt, params):
    '''Initializes the Non-uniform FFT object

    Parameters
    ----------
    kt : numpy.ndarray
        K-space trajectory
    params : dict
        Sequence parameters.

    Returns
    -------
    NufftObj : pynufft.linalg.nufft_cpu.NUFFT_cpu
        Non-uniform FFT Object for non-cartesian transformation
    '''
    kt_sc = pi / abs(np.max(kt))
    kt = kt * kt_sc # pyNUFFT scaling [-pi, pi]
    om = np.zeros((params['Npoints'] * params['Nshots'], 2))
    om[:, 0] = np.real(kt).flatten()
    om[:, 1] = np.imag(kt).flatten()

    NufftObj = pynufft.NUFFT_cpu()  # Create a pynufft object
    Nd = (params['N'], params['N'])  # image size
    Kd = (2 * params['N'], 2 * params['N'])  # k-space size
    Jd = (6, 6)  # interpolation size

    NufftObj.plan(om, Nd, Kd, Jd)
    return NufftObj