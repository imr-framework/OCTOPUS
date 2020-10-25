# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Methods to simulate different types of field maps
\nAuthor: Marina Manso Jimeno
\nLast modified: 07/16/2020
'''

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import OCTOPUS.ORC as ORC

def fieldmap_bin(field_map, bin):
    '''
    Bins a given field map given a binning value

    Parameters
    ----------
    field_map : numpy.ndarray
        Field map matrix in Hz
    bin : int
        Binning value in Hz

    Returns
    -------
    binned_field_map : numpy.ndarray
        Binned field map matrix
    '''
    fmax = field_map.max()
    bins = np.arange(-fmax, fmax + bin, bin)
    binned_field_map = np.zeros(field_map.shape)
    for x in range(field_map.shape[0]):
        for y in range(field_map.shape[1]):
            idx = ORC.find_nearest(bins, field_map[x, y])
            binned_field_map[x, y] = bins[idx]

    return binned_field_map

def parabola_formula(N):
    """
    Parabola values to fit an image of N rows/columns

    Parameters
    ----------
    N :  int
        Matrix size

    Returns
    -------
    yaxis : numpy.ndarray
        y axis values of the parabola
    """
    x1, y1 = -N / 10, 0.5
    x2, y2 = 0, 0
    x3, y3 = N / 10, 0.5

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    y = lambda x: A * x ** 2 + B * x + C
    yaxis = y(np.arange(-N / 2, N / 2, 1)).reshape(1, N)

    return yaxis

def parabolic(N, fmax, bin_opt = True, bin_val = 5):
    """
    Creates a parabolic field map

    Parameters
    ----------
    N : int
        Field map dimensions (NxN)
    fmax : float
        Frequency range in Hz
    bin_opt : bool
        Binning option. Default is True
    bin_val : int
        Binning value. Default is 5 Hz

    Returns
    -------
    field map : numpy.ndarray
        Field map matrix with dimensions [NxN] and scaled from -fmax to +fmax Hz
    """
    y = parabola_formula(N)
    rows = np.tile(y, (N,1))
    field_map_mat = rows + rows.T
    dst = np.zeros(field_map_mat.shape)
    field_map = cv2.normalize(field_map_mat, dst, -fmax, fmax, cv2.NORM_MINMAX)
    if bin_opt:
        field_map = fieldmap_bin(field_map, bin_val)

    return field_map

def hyperbolic(N, fmax, bin_opt = True,  bin_val= 5):
    """
    Creates a hyperbolic field map

    Parameters
    ----------
    N : int
       Field map dimensions (NxN)
    fmax : float
       Frequency range in Hz
    bin_opt : bool
        Binning option. Default is True
    bin_val : int
        Binning value, default is 5 Hz


    Returns
    -------
    field map : numpy.ndarray
       Field map matrix with dimensions [NxN] and scaled from -fmax to +fmax Hz
    """
    y = parabola_formula(N)
    rows = np.tile(y, (N, 1))
    field_map_mat = rows - rows.T
    dst = np.zeros(field_map_mat.shape)
    field_map = cv2.normalize(field_map_mat, dst, -fmax, fmax, cv2.NORM_MINMAX)
    if bin_opt:
        field_map = fieldmap_bin(field_map, bin_val)
    return field_map

def realistic(im , fmax , bin_opt = True, bin_val = 5):
    """
    Creates a realistic field map based on the input image

    Parameters
    ----------
    im : numpy.ndarray
        Input image
    fmax : float
       Frequency range in Hz
    bin_opt : bool
        Binning option. Default is True
    bin_val : int
        Binning value, default is 5 Hz

    Returns
    -------
    field_map : numpy.ndarray
        Field map matrix with dimensions same as im and scaled from -fmax to +fmax Hz
    """
    if im.shape[0] != im.shape[1]:
        raise ValueError('Images have to be squared (NxN)')

    N = im.shape[0]
    hist = np.histogram(im)
    mask1= cv2.threshold(im, hist[1][hist[0].argmax()+1], 1, cv2.THRESH_BINARY)
    #mask2 = cv2.threshold(im, hist[1][hist[0].argmax()], 1, cv2.THRESH_BINARY_INV)
    mask = mask1[1] #+ mask2[1]

    np.random.seed(123)
    M = np.random.rand(2, 2)
    M2 = cv2.resize(M, (N, N))


    dst = np.zeros(M2.shape)
    field_map = cv2.normalize(M2, dst, -fmax, fmax, cv2.NORM_MINMAX) * mask

    if bin_opt:
        field_map = fieldmap_bin(field_map, bin_val)

    return field_map

def spherical_order4(N, fmax, bin_opt  = True,  bin_val= 5):
    """
    Creates a field map simulating spherical harmonics of 4th order

    Parameters
    ----------
    N : int
       Field map dimensions (NxN)
    fmax : float
       Frequency range in Hz
    bin_opt : bool
        Binning option. Default is True
    bin_val : int
        Binning value, default is 5 Hz

    Returns
    -------
    field_map : numpy.ndarray
        Field map matrix with dimensions NxN and scaled from -fmax to +fmax Hz
    """
    offset = 1 #math.sqrt(1 / (4 * math.pi))
    x = np.tile(np.linspace(-offset, offset, N).reshape(N, 1), (1, N))
    l1_X = np.dstack([x] * N)
    #plot_views(l1_X, 'X')

    l1_Z = np.ones((N, N, N)) * np.linspace(-offset, offset, N).reshape(1, 1, N)
    #plot_views(l1_Z, 'Z')

    y = np.flipud(x.T)
    l1_Y = np.dstack([y] * N)
    #plot_views(l1_Y, 'Y')

    middle_slice = int(round(N / 2)) - 1
    l4_X4 = 105 * (l1_X ** 2 - l1_Y ** 2) ** 2 - 420 * l1_X ** 2 * l1_Y **2
    #plot_views(l4_X4, 'X4')
    field_map_mat = np.squeeze(l4_X4[:,:,middle_slice])

    dst = np.zeros(field_map_mat.shape)
    field_map = cv2.normalize(field_map_mat, dst, -fmax, fmax, cv2.NORM_MINMAX)
    if bin_opt:
        field_map = fieldmap_bin(field_map, bin_val)
    return field_map


'''def plot_views(matrix, fig_title):
    
    middle_slice = int(round(matrix.shape[0] / 2)) - 1
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle(fig_title, fontsize=16)
    # plt.subplot(131)
    axes[0].imshow(np.rot90(matrix[:, :, middle_slice]), vmin=matrix.min(), vmax=matrix.max())
    axes[0].set_title('Axial (XY)')
    # plt.subplot(132)
    axes[1].imshow(np.rot90(matrix[:, middle_slice, :]), vmin=matrix.min(), vmax=matrix.max())
    axes[1].set_title('Coronal (XZ)')
    # plt.subplot(133)
    axes[2].imshow(np.rot90(matrix[middle_slice, :, :]), vmin=matrix.min(), vmax=matrix.max())
    axes[2].set_title('Sagittal (YZ)')

    # plt.colorbar()
    plt.show()'''