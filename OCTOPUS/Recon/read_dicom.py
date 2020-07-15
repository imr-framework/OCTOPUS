from pydicom import dcmread
import numpy as np
import math
import matplotlib.pyplot as plt
import nibabel as nib

def read_dicom_fieldmap(path):
    data = dcmread(path)
    # Get the image data
    vol = data.pixel_array
    # Rescale from -pi to pi
    img = nib.Nifti1Image(np.rot90(vol), np.eye(4))
    nib.save(img, '../../Data/20200624/shim/phase_diff.nii.gz')
    range = data.LargestImagePixelValue + abs(data.SmallestImagePixelValue)
    newmin = -np.pi
    newmax = np.pi
    newrange = newmax + abs(newmin)
    p = (newrange / range)
    q = newmin - p * data.SmallestImagePixelValue
    vol = p * vol + q

    vol = -vol / (2 * math.pi * 2.46e-3)
    vol = np.fliplr(vol)
    plt.imshow(vol, cmap='gray')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    return vol

def read_dicom_gre(path):
    data = dcmread(path)
    vol = data.pixel_array
    vol = np.fliplr(vol)
    '''plt.imshow(vol, cmap='gray')
    plt.axis('off')
    plt.show()'''
    return vol
fpath = r'C:\Users\marin\Documents\PhD\ORC-OSSP\Data\20200624\shim\GRE_FIELD_MAPPING_0016\fieldmap.IMA'
#fpath = r'C:\Users\marin\Documents\PhD\ORC-OSSP\Data\20200624\shim\GRE_0015\TEST.MR.GEETHANATH_RECON.0015.0001.2020.06.24.17.46.43.518591.1430948.IMA'
#a = read_dicom_fieldmap(fpath)