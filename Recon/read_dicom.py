# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Methods to read DICOM files corresponding to GRE field map and GRE Siemens sequences
\nAuthor: Marina Manso Jimeno
\nLast modified: 07/16/2020
'''

from pydicom import dcmread
import numpy as np

def read_dicom(path):
    '''
    Reads dicom file from path

    Parameters
    ----------
    path : str
        Path of the file

    Returns
    -------
    vol : np.ndarray
        Array containing the image volume
    '''
    data = dcmread(path)
    # Get the image data
    vol = data.pixel_array
    return vol


