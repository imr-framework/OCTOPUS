# Copyright of the Board of Trustees of Columbia University in the City of New York

"""
Methods for ORC (Off-Resonance Correction)
"""
import numpy.fft as npfft
import numpy as np
import pynufft
import matplotlib.pyplot as plt
from math import ceil


from math import pi

def add_or(M, kt, df, nonCart = None, params = None):
    '''Adds off-resonance

    Parameters
    ----------
    M : numpy.ndarray
        Image data
    kt : numpy.ndarray
        k-space trajectory
    df : numpy.ndarray
        Field map
    nonCart : , optional
        Cartesian/Non-Cartesian trajectory option. Default is None.
    params : dict
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
            M_corr = ksp2im(kspace_orc, kt, cartesian_opt, NufftObj, params)
            M_or[x,y] = M_corr[x,y]

    return M_or

def add_or_CPR(M, kt, df, nonCart = None, params = None):
    '''Create a phase matrix - 2*pi*df*t for every df and every t'''
    if nonCart is not None:
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)

    else:
        cartesian_opt = 1
        NufftObj = None
        params = None

    kspace = im2ksp(M, cartesian_opt, NufftObj, params)

    df_values = np.unique(df)
    T = np.tile(params['t_vector'], (1, kt.shape[1]))
    M_or_CPR = np.zeros((M.shape[0], M.shape[1], len(df_values)), dtype=complex)
    kspsave= np.zeros((params['Npoints'],params['Nshots'],len(df_values)),dtype=complex)
    for i in range(len(df_values)):
        phi = 2 * pi* df_values[i] * T
        kspace_or= kspace * np.exp(-1j * phi)
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

def CPR(kspace, kt, df, nonCart = None, params = None, M = None):
    '''Off-resonance Correction by Conjugate Phase Reconstruction

    Parameters
    ----------
    kspace : numpy.ndarray
        k-space data
    kt : numpy.ndarray
        k-space trajectory
    nonCart : , optional
        Cartesian/Non-Cartesian trajectory option. Default is None.
    params : dict
        Sequence parameters. Default is None.
    M : numpy.ndarray
        Image data. Default is None.

    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''

    if nonCart is not None:
        cartesian_opt = 0
        NufftObj = nufft_init(kt, params)
    else:
        cartesian_opt = 1
        NufftObj = None
        params = None

    if M  is not None:
        kspace = im2ksp(M, cartesian_opt, NufftObj, params)

    T = np.tile(params['t_vector'], (1, kt.shape[1]))
    # kspace = fft.fftshift(fft.fft2(M))
    #kspace = im2ksp(M,kt,cartesian_opt,NufftObj,params)
    df_values = np.unique(df)
    M_CPR = np.zeros((params['N'], params['N'], len(df_values)), dtype=complex)
    for i in range(len(df_values)):
        phi = 2 * pi * df_values[i] * T
        kspace_orc = kspace* np.exp(1j * phi)
        M_corr = ksp2im(kspace_orc, cartesian_opt, NufftObj, params)
        M_CPR[:, :, i] = M_corr

    M_hat = np.zeros((params['N'],params['N']), dtype=complex)
    for x in range(df.shape[0]):
        for y in range(df.shape[1]):
            fieldmap_val = df[x,y]
            idx = np.where(df_values == fieldmap_val)
            M_hat[x,y] = M_CPR[x,y,idx]
    '''plt.imshow(abs(M_hat))
    plt.show()'''
    return M_hat

def fs_CPR(dataIn, dataInType, kt, df, Lx, params, rawData=None, M_fwd=None):
    '''Off-resonance Correction by frequency-segmented Conjugate Phase Reconstruction

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
    params : dict
        Sequence parameters. Default is None.
    M_fwd : numpy.ndarray
        Image data. Default is None.

    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''
    cartesian_opt = 0
    # pyNUFFT object
    NufftObj = nufft_init(kt, params)

    if dataInType == 'im':
        if dataIn.shape[0] != dataIn.shape[1]:
            raise ValueError('Data dimensions do not agree with expected image dimensions')
        rawData = im2ksp(dataIn, cartesian_opt, NufftObj, params)
    elif dataInType == 'raw':
        if dataIn.shape[0] != kt.shape[0] or  dataIn.shape[1] != kt.shape[1]:
            raise ValueError('Data dimensions do not agree with expected dimensions (same as ktraj)')
        rawData = dataIn
    else:
        raise ValueError('The type of input data should be either raw or im')

    # Number of frequency segments
    df_max = max(np.abs([df.max(),df.min()])) # Hz
    L = ceil(4*df_max*2*pi*params['t_readout']/pi)
    L= L * Lx
    f_L = np.linspace(df.min(),df.max(),L+1) # Hz

    T = np.tile(params['t_vector'], (1, kt.shape[1]))

    # reconstruct the L basic images
    M_fsCPR = np.zeros((params['N'],params['N'],L+1),dtype=complex)
    for l in range(L+1):
        phi = 2 * pi * f_L[l] * T
        kspace_L = rawData * np.exp(1j*phi)
        M_fsCPR[:,:,l] = ksp2im(kspace_L, cartesian_opt, NufftObj, params)

    # final image reconstruction
    M_hat = np.zeros((params['N'],params['N']),dtype=complex)
    for i in range(M_hat.shape[0]):
        for j in range(M_hat.shape[1]):
            fieldmap_val = df[i,j]
            closest_fL_idx = find_nearest(f_L, fieldmap_val)

            if fieldmap_val == f_L[closest_fL_idx]:
                pixel_val = M_fsCPR[i,j,closest_fL_idx]
            else:
                if fieldmap_val < f_L[closest_fL_idx]:
                    f_vals = [f_L[closest_fL_idx-1], f_L[closest_fL_idx]]
                    im_vals = [M_fsCPR[i,j,closest_fL_idx-1], M_fsCPR[i,j,closest_fL_idx]]
                else:
                    f_vals = [f_L[closest_fL_idx], f_L[closest_fL_idx+1]]
                    im_vals = [M_fsCPR[i, j, closest_fL_idx], M_fsCPR[i, j, closest_fL_idx+1]]

                pixel_val = np.interp(fieldmap_val, f_vals, im_vals)

            M_hat[i,j] = pixel_val

    return M_hat

def MFI(dataIn, dataInType, kt, df, Lx, params):
    '''Off-resonance Correction by Multi-Frequency Interpolation

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
    params : dict
        Sequence parameters. Default is None.
    M : numpy.ndarray
        Image data. Default is None.

    Returns
    -------
    M_hat : numpy.ndarray
        Corrected image data.
    '''
    cartesian_opt = 0

    # pyNUFFT object
    NufftObj = nufft_init(kt, params)

    if dataInType == 'im':
        if dataIn.shape[0] != dataIn.shape[1]:
            raise ValueError('Data dimensions do not agree with expected image dimensions')
        rawData = im2ksp(dataIn, cartesian_opt, NufftObj, params)
    elif dataInType == 'raw':
        if dataIn.shape[0] != kt.shape[0] or dataIn.shape[1] != kt.shape[1]:
            raise ValueError('Data dimensions do not agree with expected dimensions (same as ktraj)')
        rawData = dataIn
    else:
        raise ValueError('The type of input data should be either raw or im')


    df = np.round(df,1)
    idx, idy = np.where(df == -0.0)
    df[idx, idy] = 0.0

    # Number of frequency segments
    df_max = max(np.abs([df.max(),df.min()]))  # Hz
    df_range = (df.min(), df.max())
    L = ceil(df_max * 2 * pi * params['t_readout'] / pi)
    L = L * Lx
    f_L = np.linspace(df.min(), df.max(), L + 1)  # Hz

    T = np.tile(params['t_vector'], (1, kt.shape[1]))

    # reconstruct the L basic images
    M_MFI = np.zeros((params['N'],params['N'], L + 1), dtype=complex)
    for l in range(L + 1):
        phi = 2 * pi * f_L[l] * T
        kspace_L = rawData * np.exp(1j * phi)
        M_MFI[:, :, l] = ksp2im(kspace_L, cartesian_opt, NufftObj, params)

    # calculate MFI coefficients
    coeffs_LUT = coeffs_MFI_lsq(kt, f_L, df_range, params)

    # final image reconstruction
    M_hat = np.zeros((params['N'], params['N']), dtype=complex)
    for i in range(M_hat.shape[0]):
        for j in range(M_hat.shape[1]):
            fieldmap_val = df[i,j]
            val_coeffs = coeffs_LUT[str(fieldmap_val)]
            M_hat[i,j] = sum(val_coeffs*M_MFI[i,j,:])

    return M_hat



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
        im = npfft.ifft2(ksp)
    elif cartesian_opt == 0:
        ksp_dcf = ksp.reshape((params['Npoints']*params['Nshots'],))*params['dcf']
        im = NufftObj.adjoint(ksp_dcf)

        #im = NufftObj.solve(ksp_dcf, solver='cg', maxiter=50)
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

    om = np.zeros((params['Npoints'] * params['Nshots'], 2))
    om[:, 0] = np.real(kt).flatten()
    om[:, 1] = np.imag(kt).flatten()

    NufftObj = pynufft.NUFFT_cpu()  # Create a pynufft object
    Nd = (params['N'], params['N'])  # image size
    Kd = (2 * params['N'], 2 * params['N'])  # k-space size
    Jd = (6, 6)  # interpolation size

    NufftObj.plan(om, Nd, Kd, Jd)
    return NufftObj

def find_nearest(array,value):
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
    idx = (np.abs(array - value)).argmin()
    return idx

def coeffs_MFI_lsq(kt, f_L, df_range,params):
    '''Finds the coefficients for Multi-frequency interpolation method by least squares approximation.

    Parameters
    ----------
    kt : numpy.ndarray
        K-space trajectory
    f_L : numpy.ndarray
        Frequency segments array.
    df_range :
        Frequency range of the field map (minimum and maximum).
    params : dict
        Sequence parameters.

    Returns
    -------
    cL : dict
        Coefficients look-up-table.
    '''
    fs = 0.1 # Hz
    f_sampling = np.round(np.arange(df_range[0], df_range[1]+fs, fs),1)


    T = params['t_vector'][:,0] # specific to siemens, might have to change it. Also consider starting at TE

    A = np.zeros((kt.shape[0], f_L.shape[0]), dtype=complex)
    for l in range(f_L.shape[0]):
        phi = 2*pi*f_L[l]*T
        A[:, l] = np.exp(-1j *phi)

    cL={}
    for fs in f_sampling:
        b = np.exp(-1j* 2*pi*fs*T)
        C = np.linalg.lstsq(A, b, rcond=None)
        if fs == -0.0:
            fs = 0.0
        cL[str(fs)] = C[0][:].reshape((f_L.shape[0]))

    return cL

def polynomial_fit(df,M):
    ''' Deprecated '''
    S = 0
    S_x = 0
    S_y = 0
    S_f =0
    S_xx = 0
    S_yy = 0
    S_xy = 0
    S_xf = 0
    S_yf = 0
    for xi in range(M.shape[0]):
        for yi in range(M.shape[1]):
            omega = 1/np.abs(M[xi, yi])
            S = 1/omega**2
            S_x = xi/omega**2
            S_y = yi/omega**2
            S_f = df[xi,yi]/omega**2
            S_xx = xi**2/omega**2
            S_yy = yi ** 2 / omega ** 2
            S_xy = xi*yi/omega**2
            S_xf = xi * df[xi,yi] / omega ** 2
            S_yf = yi*df[xi,yi] / omega ** 2

        S += S
        S_x += S_x
        S_y += S_y
        S_f += S_f
        S_xx += S_xx
        S_yy += S_yy
        S_xy += S_xy
        S_xf += S_xf
        S_yf += S_yf

    delta = np.asarray(([S, S_x, S_y], [S_x, S_xx, S_xy],[S_y, S_xy, S_yy]))
    delta_f = np.asarray(([S_f, S_x, S_y],[S_xf, S_xx, S_xy], [S_yf, S_xy, S_yy]))
    delta_x = np.linalg.det(np.asarray(([S, S_f, S_y],[S_x, S_xf, S_xy], [S_y, S_yf, S_yy])))
    delta_y = np.linalg.det(np.asarray(([S, S_x, S_f], [S_x, S_xx, S_xf], [S_y, S_xy, S_yf])))


    f_0 = delta_f/delta
    alpha = delta_x/delta
    beta = delta_y/delta

    lin_df = np.zeros(df.shape)
    for xi in range(df.shape[0]):
        for yi in range(df.shape[1]):
            lin_df[xi,yi] = f_0+alpha*xi+beta*yi


    return lin_df