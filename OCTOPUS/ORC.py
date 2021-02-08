# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Methods for ORC (Off-Resonance Correction) and off-resonance simulation
\nAuthor: Marina Manso Jimeno
\nLast updated: 01/20/2021
'''

import numpy.fft as npfft
import numpy as np

from math import ceil, pi

from OCTOPUS.recon.imtransforms import ksp2im, im2ksp, nufft_init


def check_inputs_cartesian(dataInShape, dataInType, ktShape, dfShape):
    '''Check dimensions of the inputs

    Parameters
    ----------
    dataInShape :  tuple
        Shape of input data. Must be [NxN] for image data and [NlinesxNcolumns] for raw data.
    dataInType : str
        Type of data: im or raw
    ktShape : tuple
        Shape of k-space trajectory. Must be [NlinesxNcolumns].
    dfShape : tuple
        Shape of field map. Must be [NxN] and match the image matrix size.
    '''
    if dataInType == 'im':
        if dataInShape[0] != dataInShape[1]:
            raise ValueError('Data dimensions do not agree with expected image dimensions (NxN)')
        if dataInShape != dfShape:
            raise ValueError('The image and field map dimensions do not match')
    elif dataInType == 'raw':
        if dataInShape != ktShape:
            raise ValueError('The raw data and k-space trajectory dimensions do not match')
    else:
        raise ValueError('This type of input data is not supported. Please select im or raw')


def check_inputs_noncartesian(dataInShape, dataInType, ktShape, dfShape, params):
    '''Check dimensions of the inputs

    Parameters
    ----------
    dataInShape :  tuple
        Shape of input data. Must be [NxN] for image data and [NpointsxNshots] for raw data.
    dataInType : str
        Type of data: im or raw
    ktShape : tuple
        Shape of k-space trajectory. Must be [NpointsxNshots].
    dfShape : tuple
        Shape of field map. Must be [NxN] and match the image matrix size.
    params : dict
        Sequence parameters. Required elements are Npoints, Nshots, and N.
    '''
    if dataInType == 'im':
        if dataInShape[0] != dataInShape[1]:
            raise ValueError('Data dimensions do not agree with expected image dimensions (NxN)')
        if dataInShape != dfShape:
            raise ValueError('The image and field map dimensions do not match')

    elif dataInType == 'raw':
        if dataInShape != ktShape:
            raise ValueError('The raw data and k-space trajectory dimensions do not match')

    else:
        raise ValueError('This type of input data is not supported. Please select im or raw')

    if 'Npoints' not in params:
        raise ValueError('The number of acquisition points is missing')
    if 'Nshots' not in params:
        raise ValueError('The number of shots is missing')
    if 'N' not in params:
        raise ValueError('The matrix size N is missing')

    if params['N'] != dfShape[0]:
        raise ValueError('The image or field map dimensions and value specified for N do not match')

    if params['Npoints'] != ktShape[0]:
        raise ValueError('The raw data or trajectory dimensions and value specified for Npoints do not match')

    if params['Nshots'] != ktShape[1]:
        raise ValueError('The raw data or trajectory dimensions and value specified for Nshots do not match')




def add_or(M, kt, df, nonCart = None, params = None):
    '''Forward model for off-resonance simulation

    Parameters
    ----------
    M : numpy.ndarray
        Image data
    kt : numpy.ndarray
        k-space trajectory
    df : numpy.ndarray
        Field map
    nonCart : int , optional
        Cartesian/Non-Cartesian trajectory option. Default is None.
    params : dict , optional
        Sequence parameters. Default is None.

    Returns
    -------
    M_or : numpy.ndarray
        Off-resonance corrupted image data
    '''

    '''Create a phase matrix - 2*pi*df*t for every df and every t'''
    if nonCart is not None:
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)

    else:
        cartesian_opt = 1
        NufftObj = None
        params = None

    kspace = im2ksp(M, cartesian_opt, NufftObj, params)
    M_or = np.zeros(M.shape, dtype=complex)

    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            phi = 2*pi*df[x,y]*kt
            kspace_orc = kspace*np.exp(-1j*phi)
            M_corr = ksp2im(kspace_orc, cartesian_opt, NufftObj, params)
            M_or[x,y] = M_corr[x,y]

    return M_or

def add_or_CPR(M, kt, df, nonCart = None, params = None):
    '''Forward model for off-resonance simulation. The number of fourier transforms = number of unique values in the field map.

    Parameters
    ----------
    M : numpy.ndarray
        Image data
    kt : numpy.ndarray
        k-space trajectory
    df : numpy.ndarray
        Field map
    nonCart : int , optional
        Cartesian/Non-Cartesian trajectory option. Default is None.
    params : dict , optional
        Sequence parameters. Default is None.

    Returns
    -------
    M_or : numpy.ndarray
        Off-resonance corrupted image data
    '''
    # Create a phase matrix - 2*pi*df*t for every df and every t
    if nonCart is not None and nonCart == 1:
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)
        T = np.tile(params['t_vector'], (1, kt.shape[1]))

    elif nonCart == 'EPI':
        cartesian_opt = 1
        NufftObj = None
        T = np.flipud(params['t_vector']).reshape(params['N'], params['N'])
        T[1:params['N']:2,:] = np.fliplr(T[1:params['N']:2,:])

    else:
        cartesian_opt = 1
        NufftObj = None
        params = None
        T = kt

    kspace = im2ksp(M, cartesian_opt, NufftObj, params)

    df_values = np.unique(df)

    M_or_CPR = np.zeros((M.shape[0], M.shape[1], len(df_values)), dtype=complex)
    kspsave = np.zeros((kspace.shape[0],kspace.shape[1],len(df_values)),dtype=complex)
    for i in range(len(df_values)):
        phi = - 2 * pi* df_values[i] * T
        kspace_or = kspace * np.exp(1j * phi)
        kspsave[:,:,i] = kspace_or
        M_corr = ksp2im(kspace_or, cartesian_opt, NufftObj, params)
        M_or_CPR[:, :, i] = M_corr

    M_or = np.zeros(M.shape, dtype=complex)
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            fieldmap_val = df[x, y]
            idx = np.where(df_values == fieldmap_val)
            M_or[x, y] = M_or_CPR[x, y, idx]
    '''plt.imshow(abs(M_or))
    plt.show()'''
    return M_or, kspsave

def orc(M, kt, df):
    '''Off-resonance correction for Cartesian trajectories

    Parameters
    ----------
    M : numpy.ndarray
        Cartesian k-space data
    kt : numpy.ndarray
        Cartesian k-space trajectory
    df : numpy.ndarray
        Field map

    Returns
    -------
    M_hat : numpy.ndarray
        Off-resonance corrected image data
    '''

    kspace = npfft.fftshift(npfft.fft2(M))
    M_hat = np.zeros(M.shape, dtype=complex)
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            phi = 2 * pi * df[x, y] * kt
            kspace_orc = kspace * np.exp(1j * phi)
            M_corr = npfft.ifft2(kspace_orc)
            M_hat[x, y] = M_corr[x, y]

    return M_hat

def CPR(dataIn, dataInType, kt, df, nonCart=None, params=None):
    '''Off-resonance Correction by Conjugate Phase Reconstruction
    Maeda, A., Sano, K. and Yokoyama, T. (1988), Reconstruction by weighted correlation for MRI with time-varying gradients. IEEE Transactions on Medical Imaging, 7(1): 26-31. doi: 10.1109/42.3926

    Parameters
    ----------
    dataIn : numpy.ndarray
        k-space raw data or image data
    dataInType : str
        Can be either 'raw' or 'im'
    kt : numpy.ndarray
        k-space trajectory.
    df : numpy.ndarray
        Field map
    nonCart : int
        Non-cartesian trajectory option. Default is None (Cartesian).
    params : dict
        Sequence parameters. Default is None (Cartesian).

    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''

    if nonCart is not None and nonCart == 1:
        check_inputs_noncartesian(dataIn.shape, dataInType, kt.shape, df.shape, params)
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)
        T = np.tile(params['t_vector'], (1, kt.shape[1]))
        N = params['N']

    elif nonCart == 'EPI':
        # check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        T = np.flipud(params['t_vector']).reshape(params['N'], params['N'])
        T[1:params['N']:2,:] = np.fliplr(T[1:params['N']:2,:])

        N = dataIn.shape[0]

    else:
        check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        T = kt
        N = dataIn.shape[0]

    if dataInType == 'im':
        rawData = im2ksp(dataIn, cartesian_opt, NufftObj, params)
    elif dataInType == 'raw':
        rawData = dataIn

    df_values = np.unique(df)
    M_CPR = np.zeros((N, N, len(df_values)), dtype=complex)
    for i in range(len(df_values)):
        phi = 2 * pi * df_values[i] * T
        kspace_orc = rawData * np.exp(1j * phi)
        M_corr = ksp2im(kspace_orc, cartesian_opt, NufftObj, params)
        M_CPR[:, :, i] = M_corr

    M_hat = np.zeros((N, N), dtype=complex)
    for x in range(df.shape[0]):
        for y in range(df.shape[1]):
            fieldmap_val = df[x,y]
            idx = np.where(df_values == fieldmap_val)
            M_hat[x,y] = M_CPR[x,y,idx]

    return M_hat

def fs_CPR(dataIn, dataInType, kt, df, Lx, nonCart= None, params= None):
    '''Off-resonance Correction by frequency-segmented Conjugate Phase Reconstruction
    Noll, D. C., Pauly, J. M., Meyer, C. H., Nishimura, D. G. and Macovskj, A. (1992), Deblurring for non‐2D fourier transform magnetic resonance imaging. Magn. Reson. Med., 25: 319-333. doi:10.1002/mrm.1910250210

    Parameters
    ----------
    dataIn : numpy.ndarray
        k-space raw data or image data
    dataInType : str
        Can be either 'raw' or 'im'
    kt : numpy.ndarray
        k-space trajectory
    df : numpy.ndarray
        Field map
    Lx : int
        L (frequency bins) factor
    nonCart : int
        Non-cartesian trajectory option. Default is None (Cartesian).
    params : dict
        Sequence parameters. Default is None (Cartesian).

    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''
    if nonCart is not None and nonCart == 1:
        check_inputs_noncartesian(dataIn.shape, dataInType, kt.shape, df.shape, params)

        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)
        T = np.tile(params['t_vector'], (1, kt.shape[1]))

        N = params['N']
        t_ro = T[-1, 0] - T[0, 0]  # T[end] - TE

    elif nonCart == 'EPI':
        # check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        T = np.flipud(params['t_vector']).reshape(params['N'], params['N'])
        T[1:params['N']:2,:] = np.fliplr(T[1:params['N']:2,:])
        N = dataIn.shape[0]
        t_ro = params['t_readout']


    else:
        check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        N = dataIn.shape[0]

        t_vector = kt[0].reshape(kt.shape[1], 1)
        T = kt
        t_ro = T[0, -1] - T[0, 0]

    if dataInType == 'im':
        rawData = im2ksp(dataIn, cartesian_opt, NufftObj, params)
    elif dataInType == 'raw':
        rawData = dataIn

    # Number of frequency segments
    df_max = max(np.abs([df.max(), df.min()])) # Hz
    L = ceil(4 * df_max * 2 * pi * t_ro / pi)
    L= L * Lx
    if len(np.unique(df)) == 1:
        L = 1
    f_L = np.linspace(df.min(), df.max(), L + 1) # Hz

    # T = np.tile(params['t_vector'], (1, kt.shape[1]))

    # reconstruct the L basic images
    M_fsCPR = np.zeros((N, N, L + 1),dtype=complex)
    for l in range(L+1):
        phi = 2 * pi * f_L[l] * T
        kspace_L = rawData * np.exp(1j * phi)
        M_fsCPR[:,:,l] = ksp2im(kspace_L, cartesian_opt, NufftObj, params)

    # final image reconstruction
    M_hat = np.zeros((N, N), dtype=complex)
    for i in range(M_hat.shape[0]):
        for j in range(M_hat.shape[1]):
            fieldmap_val = df[i, j]
            closest_fL_idx = find_nearest(f_L, fieldmap_val)

            if fieldmap_val == f_L[closest_fL_idx]:
                pixel_val = M_fsCPR[i, j, closest_fL_idx]
            else:
                if fieldmap_val < f_L[closest_fL_idx]:
                    f_vals = [f_L[closest_fL_idx - 1], f_L[closest_fL_idx]]
                    im_vals = [M_fsCPR[i, j, closest_fL_idx - 1], M_fsCPR[i, j, closest_fL_idx]]
                else:
                    f_vals = [f_L[closest_fL_idx], f_L[closest_fL_idx + 1]]
                    im_vals = [M_fsCPR[i, j, closest_fL_idx], M_fsCPR[i, j, closest_fL_idx + 1]]

                pixel_val = np.interp(fieldmap_val, f_vals, im_vals)

            M_hat[i, j] = pixel_val

    return M_hat

def MFI(dataIn, dataInType, kt , df, Lx , nonCart= None, params= None):
    '''Off-resonance Correction by Multi-Frequency Interpolation
    Man, L., Pauly, J. M. and Macovski, A. (1997), Multifrequency interpolation for fast off‐resonance correction. Magn. Reson. Med., 37: 785-792. doi:10.1002/mrm.1910370523

    Parameters
    ----------
    dataIn : numpy.ndarray
        k-space raw data or image data
    dataInType : str
        Can be either 'raw' or 'im'
    kt : numpy.ndarray
        k-space trajectory
    df : numpy.ndarray
        Field map
    Lx : int
        L (frequency bins) factor
    nonCart : int
        Non-cartesian trajectory option. Default is None (Cartesian).
    params : dict
        Sequence parameters. Default is None.
    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''

    if nonCart is not None and nonCart == 1:
        check_inputs_noncartesian(dataIn.shape, dataInType, kt.shape, df.shape, params)
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)
        t_vector = params['t_vector']
        T = np.tile(params['t_vector'], (1, kt.shape[1]))
        t_ro = T[-1,0] - T[0,0] # T[end] - TE
        N = params['N']

    elif nonCart == 'EPI':
        # check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        T = np.flipud(params['t_vector']).reshape(params['N'], params['N'])
        T[1:params['N']:2,:] = np.fliplr(T[1:params['N']:2,:])
        N = dataIn.shape[0]
        t_ro = params['t_readout']
        t_vector = params['t_vector']

    else:
        check_inputs_cartesian(dataIn.shape, dataInType, kt.shape, df.shape)
        cartesian_opt = 1
        NufftObj = None
        N = dataIn.shape[0]

        t_vector = kt[0].reshape(kt.shape[1],1)
        T = kt
        t_ro = T[0, -1] - T[0,0]

    if dataInType == 'im':
        rawData = im2ksp(dataIn, cartesian_opt, NufftObj, params)
    elif dataInType == 'raw':
        rawData = dataIn

    df = np.round(df, 1)
    idx, idy = np.where(df == -0.0)
    df[idx, idy] = 0.0

    # Number of frequency segments
    df_max = max(np.abs([df.max(), df.min()]))  # Hz
    df_range = (df.min(), df.max())
    L = ceil(df_max * 2 * pi * t_ro / pi)
    L = L * Lx
    if len(np.unique(df)) == 1:
        L = 1
    f_L = np.linspace(df.min(), df.max(), L + 1)  # Hz

    #T = np.tile(params['t_vector'], (1, kt.shape[1]))

    # reconstruct the L basic images
    M_MFI = np.zeros((N, N, L + 1), dtype=complex)
    for l in range(L + 1):
        phi = 2 * pi * f_L[l] * T
        kspace_L = rawData * np.exp(1j * phi)
        M_MFI[:, :, l] = ksp2im(kspace_L, cartesian_opt, NufftObj, params)

    # calculate MFI coefficients
    coeffs_LUT = coeffs_MFI_lsq(kt, f_L, df_range, t_vector)

    # final image reconstruction
    M_hat = np.zeros((N, N), dtype=complex)
    for i in range(M_hat.shape[0]):
        for j in range(M_hat.shape[1]):
            fieldmap_val = df[i,j]
            val_coeffs = coeffs_LUT[str(fieldmap_val)]
            M_hat[i, j] = sum(val_coeffs * M_MFI[i, j, :])

    return M_hat

def find_nearest(array, value):
    '''Finds the index of the value's closest array element

    Parameters
    ----------
    array : numpy.ndarray
        Array of values
    value : float
        Value for which the closest element has to be found

    Returns
    -------
    idx : int
        Index of the closest element of the array
    '''
    array = np.asarray(array)
    diff = array - value
    if value >= 0:
        idx = np.abs(diff).argmin()
    else:
        idx = np.abs(diff[array < 1]).argmin()

    return idx

def coeffs_MFI_lsq(kt, f_L, df_range, t_vector):
    '''Finds the coefficients for Multi-frequency interpolation method by least squares approximation.

    Parameters
    ----------
    kt : numpy.ndarray
        K-space trajectory
    f_L : numpy.ndarray
        Frequency segments array.
    df_range : tuple
        Frequency range of the field map (minimum and maximum).
    params : dict
        Sequence parameters.

    Returns
    -------
    cL : dict
        Coefficients look-up-table.
    '''
    fs = 0.1 # Hz
    f_sampling = np.round(np.arange(df_range[0], df_range[1]+fs, fs), 1)

    alpha = 1.2 #
    t_limit = t_vector[-1]

    T = np.linspace(0, alpha * t_limit, len(t_vector)).reshape(-1, )

    A = np.zeros((len(t_vector), f_L.shape[0]), dtype=complex)
    for l in range(f_L.shape[0]):
        phi = 2 * pi * f_L[l] * T
        A[:, l] = np.exp(1j * phi)

    cL={}
    for fs in f_sampling:
        b = np.exp(1j * 2 * pi * fs * T)
        C = np.linalg.lstsq(A, b, rcond=None)
        if fs == -0.0:
            fs = 0.0
        cL[str(fs)] = C[0][:].reshape((f_L.shape[0]))

    return cL
