# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Author: Marina Manso Jimeno
Last modified: 07/21/2020
'''
import numpy as np
import scipy.io as sio
import nibabel as nib
import os

def get_data_from_file(input):
    '''
    Get the data of a file given a string or a dictionary independently of the data format

    Parameters
    ----------
    input : str or dict
        Input path or dictionary in case of user upload (Colab)

    Returns
    -------
    file_data : np.ndarray
        Data extracted from the file
    '''
    if isinstance(input, str):
        file_name = input
    elif isinstance(input, dict):
        file_name = next(iter(input))
    file_format = file_name[file_name.find('.'):]
    if file_format == '.mat':
        file_data_dict = sio.loadmat(file_name)
        file_data = file_data_dict[list(file_data_dict.keys())[-1]]
    elif file_format == '.npy':
        file_data = np.load(file_name)
    elif file_format == '.nii.gz':
        file_data = nib.load(file_name).get_fdata()
    else:
        raise ValueError('Sorry, this file format is not supported:' + (file_format))

    return file_data

from pydicom import dcmread


def read_dicom(path):
    '''
    Reads dicom file from path

    Parameters
    ----------
    path : str
        Path of the file/dicom folder

    Returns
    -------
    vol : np.ndarray
        Array containing the image volume
    '''
    vol = []
    if os.path.isdir(path):
        for files_in in os.listdir(path):
            vol.append(dcmread(os.path.join(path, files_in)).pixel_array)
        vol = np.moveaxis(np.stack(vol), 0, -1)
    elif os.path.isfile(path):
        data = dcmread(path)
    # Get the image data
        vol = data.pixel_array
    return vol


