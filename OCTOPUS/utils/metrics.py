'''
Methods to calculate off-resonance correction performance metrics
Author: Marina Manso Jimeno
Last updated: 06/03/2020
'''

import scipy.io as sio
import cv2

from skimage.metrics import peak_signal_noise_ratio as pSNR
from skimage.metrics import structural_similarity as SSIM


import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float

def SSIM_calc(im1 : np.ndarray, im2 : np.ndarray):
    '''
    Structural Similarity Index Calculation for two images

    Parameters
    ----------
    im1 : nupy.ndarray
        Image 1. Reference image, same shape as im2
    im2 : numpy.ndarray
        Image 2

    Returns
    -------
    hfen : float
        Measured SSIM value
    '''
    ssim = SSIM(im1, im2, data_range=im2.max() - im2.min())
    return ssim

def pSNR_calc(im1,im2):
    '''
    Peak Signal to Noise Ratio Calculation for two images

    Parameters
    ----------
    im1 : nupy.ndarray
        Image 1. Reference image, same shape as im2
    im2 : numpy.ndarray
        Image 2

    Returns
    -------
    hfen : float
        Measured SSIM value
    '''
    psnr = pSNR(im1, im2, data_range=im2.max() - im2.min())
    return psnr

def LoG(im):
    '''
    Laplacian of a Gaussian implementation with kernel size (15, 15) and sigma equal to 1.5 pixels

    Parameters
    ----------
    im : numpy.ndarray
        Input image

    Returns
    -------
    log_im : numpy.ndarray
        Laplacian of a gaussian of the image
    '''
    # rotationally symmetric kernel
    log_kernel = sio.loadmat('log_kernel.mat')['log_kernel']
    log_im = cv2.filter2D(np.abs(im), -1, log_kernel)

    return log_im

def HFEN(im1 : np.ndarray, im2 : np.ndarray):
    '''
    High Frequency Error Norm calculation for two images

    Parameters
    ----------
    im1 : nupy.ndarray
        Image 1. Reference image, same shape as im2
    im2 : numpy.ndarray
        Image 2

    Returns
    -------
    hfen : float
        Measured HFEN value
    '''
    log1 = LoG(im1)
    log2 = LoG(im2)

    hfen = np.linalg.norm((log1 - log2), 2) / np.linalg.norm(log2, 2)

    return hfen

def create_table(stack_of_images : np.ndarray, col_names : tuple, franges : tuple):
    '''
    Displays a table with the metrics for images corrected using the different ORC methods

    Parameters
    ----------
    stack_of_images : numpy.ndarray
        stack_of_images[0] : ground truth (before correction). stack_of_images[1] : CPR corrected. stack_of_images[2] : fs-CPR corrected. stack_of_images[3] : MFI corrected.
    col_names : tuple
        Names for the columns
    franges :  tuple
        Frequency ranges of the original field map
    '''

    nmetrics = 3
    nmethods = len(stack_of_images) - 1

    if len(col_names) != nmethods:
        raise ValueError('Number of methods does not match the number of columns for the table')
    if isinstance(franges, tuple):
        nfranges = len(franges)
    else:
        franges = [franges]
        nfranges = 1
        stack_of_images = np.expand_dims(im_stack, axis=3)

    data = np.zeros((nfranges, nmetrics, nmethods))

    for fr in range(nfranges):
        ims_fr = np.squeeze(stack_of_images[:,:,:,fr])
        GT = ims_fr[0]
        for col in range(nmethods):
            corr_im = ims_fr[col + 1]
            data[fr, 0, col] = pSNR(GT, corr_im, data_range=corr_im.max() - corr_im.min())
            data[fr, 1, col] = SSIM(GT, corr_im, data_range=corr_im.max() - corr_im.min())
            data[fr, 2, col] = HFEN(GT, corr_im)

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0.25, 0.7, nmetrics))
    labels = ['pSNR', 'SSIM', 'HFEN']
    fig, ax = plt.subplots(nmetrics, 1, sharex=True)
    x = range(nmethods)
    plt.xticks(x, col_names)

    for fr in range(nfranges):
        data_fr = np.squeeze(data[fr,:,:])
        for row in range(nmetrics):
            ax[row].plot(x, data_fr[row, :], 'o-', color=colors[fr], label=franges[fr] if row == 0 else "")
            ax[row].set_ylabel(labels[row])
            if row != nmetrics -1:
                ax[row].xaxis.set_visible(False)
    fig.legend()

    if nfranges == 1:
        table_data = np.round(data[0],2)
    else:

        #col_names= list(col_names)
        #col_names.insert(0, 'Freq range')
        table_data = np.zeros((nfranges * nmetrics, nmethods )).tolist()
        row_count = 0
        row_labels = []
        row_colors = []
        for metric in range(nmetrics):
            for fr in range(nfranges):
                row_labels.append(franges[fr])
                row_colors.append(colors[fr])
                table_data[row_count][:] = np.round(data[fr, metric, :],2)
                row_count += 1

    the_table = plt.table(cellText = table_data,
                          cellLoc= 'center',
                          rowLabels=row_labels,
                          rowColours=row_colors,
                          colLabels=col_names,
                          loc='bottom')

    h = the_table.get_celld()[(0, 0)].get_height()
    w = the_table.get_celld()[(0, 0)].get_width() / (nmethods + 1)
    header = [the_table.add_cell(pos, -2, w, h, loc="center", facecolor="none") for pos in range(1, nfranges * nmetrics + 1)]
    count = 0
    for i in range(0, nfranges * nmetrics, nmetrics):
        header[i].visible_edges = "TLR"
        header[i+1].visible_edges = "LR"
        header[i+2].visible_edges = "LR"
        header[i + 1].get_text().set_text(labels[count])
        count += 1
    header[-1].visible_edges = 'BLR'

    the_table._bbox = [0, -2, 1, 1.7]
    the_table.set_fontsize(8)
    the_table.scale(1.5, 1.5)
    plt.subplots_adjust(bottom=0.35, left=0.2)
    fig.suptitle('Performance metrics')
    plt.show()


'''ph = np.load('../Recon/test_data/slph_im.npy') # Shep-Logan Phantom
#ph = sio.loadmat('test_data/sl_ph.mat')['sl_ph']
ph = (ph - np.min(ph)) / (np.max(ph)-np.min(ph))
ph1 = cv2.GaussianBlur(ph, (99,99),0)
plt.subplot(121)
plt.imshow(ph)
plt.subplot(122)
plt.imshow(ph1)
plt.show()
a = HFEN(ph1, ph)'''
'''img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1
img_noise = img + noise
img_const = img + abs(noise)
im_stack = np.stack((img, img_noise, img_const))
cols = ('Image with noise', 'Image plus constant')

create_table(im_stack, cols)'''
'''im_stack = np.load('../Recon/im_stack22.npy')
cols = ('CPR', 'fs-CPR', 'MFI')
f_ranges = ('-/+ 250 Hz', '-/+ 500 Hz', '-/+ 750 Hz')'''

'''im_stack = np.load('../Recon/im_stack.npy')
cols = ('CPR', 'fs-CPR', 'MFI')
f_ranges = ('-/+ 250 Hz')'''

'''create_table(im_stack, cols, f_ranges)'''