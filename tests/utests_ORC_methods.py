# Copyright of the Board of Trustees of Columbia University in the City of New York
import unittest

import scipy.io as sio
import numpy as np

import OCTOPUS.ORC as ORC

from OCTOPUS.fieldmap.simulate import fieldmap_bin

# Inputs
raw_data = sio.loadmat('test_data/acrph_noncart_ksp.mat')['dat']
ktraj = np.load('test_data/ktraj_noncart.npy')
ktraj_dcf = np.load('test_data/ktraj_noncart_dcf.npy').flatten()
t_ro = ktraj.shape[0] * 10e-6
T = np.linspace(4.6e-3, 4.6e-3 + t_ro, ktraj.shape[0]).reshape((ktraj.shape[0], 1))

field_map = np.load('test_data/acrph_df.npy')
field_map_CPR = fieldmap_bin(field_map, 5)
acq_params = {'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 'N': field_map.shape[0], 'dcf': ktraj_dcf, 't_vector': T, 't_readout': t_ro}


class TestfscPR(unittest.TestCase):
    def test_fsCPR_given_rawData(self):
        # Test correction given raw data
        or_corr_im = ORC.fs_CPR(dataIn=raw_data[:,:,0], dataInType='raw', kt=ktraj, df=field_map, Lx=1, nonCart=1, params=acq_params)
        self.assertEqual(or_corr_im.shape, field_map.shape[:-1]) # Dimensions agree

    def test_fsCPR_given_im(self):
        # Test correction given image data
        or_corr_im = ORC.fs_CPR(dataIn=raw_data[:,:,0],dataInType='raw',kt=ktraj,df=field_map,Lx=1, nonCart=1, params=acq_params)
        or_corr_im2 = ORC.fs_CPR(dataIn=or_corr_im,dataInType='im',kt=ktraj,df=np.squeeze(field_map),Lx=1,nonCart=1,params=acq_params)
        self.assertEqual(or_corr_im2.shape, field_map.shape[:-1]) # Dimensions match
        self.assertEqual(or_corr_im2.all(), or_corr_im.all())

    def test_fsCPR_wrongtype(self):
        # Error when dataIn is raw data but dataInType is 'im'
        with self.assertRaises(ValueError): ORC.fs_CPR(dataIn=raw_data[:, :, 0], dataInType='im', kt=ktraj, df=field_map, Lx=1, nonCart=1, params=acq_params)
        or_corr_im = ORC.fs_CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map, Lx=1,
                                nonCart=1, params=acq_params)
        # Error when dataIn is image data but dataInType is 'raw'
        with self.assertRaises(ValueError): ORC.fs_CPR(dataIn=or_corr_im, dataInType='raw', kt=ktraj,
                                                       df=field_map, Lx=1, nonCart=1, params=acq_params)
        # Error when dataInType is other than 'raw' or 'im'
        with self.assertRaises(ValueError): ORC.fs_CPR(dataIn=raw_data[:, :, 0], dataInType='other', kt=ktraj,
                                                       df=field_map, Lx=1, nonCart=1, params=acq_params)

    def test_find_nearest(self):
        # test that find_nearest works for positive and negative values
        array = np.linspace(-10, 10, 21)
        neg_val = -5
        pos_val = 5
        idx_neg = ORC.find_nearest(array, neg_val)
        idx_pos = ORC.find_nearest(array, pos_val)
        self.assertEqual(array[idx_neg], neg_val)
        self.assertEqual(array[idx_pos], pos_val)

class TestMFI(unittest.TestCase):
    def test_MFI_given_rawData(self):
        # Test correction given raw data
        or_corr_im = ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map[:,:,0], Lx=1,
                                nonCart=1, params=acq_params)
        self.assertEqual(or_corr_im.shape, field_map.shape[:-1])  # Dimensions agree

    def test_MFI_given_im(self):
        # Test correction given image data
        or_corr_im = ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map[:,:,0], Lx=1,
                                nonCart=1, params=acq_params)
        or_corr_im2 = ORC.MFI(dataIn=or_corr_im, dataInType='im', kt=ktraj, df=field_map[:,:,0], Lx=1, nonCart=1, params=acq_params)
        self.assertEqual(or_corr_im2.shape, field_map.shape[:-1])
        self.assertEqual(or_corr_im2.all(), or_corr_im.all())

    def test_MFI_wrongtype(self):
        # Error when dataIn is raw data but dataInType is 'im'
        with self.assertRaises(ValueError): ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='im', kt=ktraj,
                                                       df=field_map, Lx=1, nonCart=1, params=acq_params)
        or_corr_im = ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map[:,:,0], Lx=1, nonCart=1,
                                params=acq_params)
        # Error when dataIn is image data but dataInType is 'raw'
        with self.assertRaises(ValueError): ORC.MFI(dataIn=or_corr_im, dataInType='raw', kt=ktraj,
                                                       df=field_map, Lx=1, nonCart=1, params=acq_params)
        # Error when dataInType is other than 'raw' or 'im'
        with self.assertRaises(ValueError): ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='other', kt=ktraj,
                                                       df=field_map, Lx=1, nonCart=1, params=acq_params)


class testCPR(unittest.TestCase):
    def test_CPR_given_rawData(self):
        # Test correction given raw data
        or_corr_im = ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map_CPR[:,:,0],
                                nonCart=1, params=acq_params)
        self.assertEqual(or_corr_im.shape, field_map.shape[:-1])  # Dimensions agree

    def test_CPR_given_im(self):
        # Test correction given image data
        or_corr_im = ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map_CPR[:,:,0],
                                nonCart=1, params=acq_params)
        or_corr_im2 = ORC.CPR(dataIn=or_corr_im, dataInType='im', kt=ktraj, df=field_map_CPR[:,:,0], nonCart=1, params=acq_params)
        self.assertEqual(or_corr_im2.shape, field_map.shape[:-1])
        self.assertEqual(or_corr_im2.all(), or_corr_im.all())

    def test_CPR_wrongtype(self):
        # Error when dataIn is raw data but dataInType is 'im'
        with self.assertRaises(ValueError): ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='im', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=acq_params)
        or_corr_im = ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj, df=field_map_CPR[:,:,0], nonCart=1,
                                params=acq_params)
        # Error when dataIn is image data but dataInType is 'raw'
        with self.assertRaises(ValueError): ORC.CPR(dataIn=or_corr_im, dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=acq_params)
        # Error when dataInType is other than 'raw' or 'im'
        with self.assertRaises(ValueError): ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='other', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=acq_params)

    def test_CPR_wrongInputDimensions_imdata(self):
        # Image data is not NxN
        data = np.ones((acq_params['N'],acq_params['N']+1))
        with self.assertRaises(ValueError): ORC.CPR(dataIn=data, dataInType='im', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=acq_params) # non-Cartesian
        with self.assertRaises(ValueError): ORC.CPR(dataIn=data, dataInType='im', kt=ktraj,
                                                       df=field_map_CPR) # cartesian

    def test_CPR_wrongInputDimensions_rawdata(self):
        # raw data dimensions do not match ktraj dimensions
        data = np.ones((acq_params['N'], acq_params['N'] + 1))
        with self.assertRaises(ValueError): ORC.CPR(dataIn=data, dataInType='raw', kt=ktraj,
                                                    df=field_map_CPR)  # cartesian


class testParametersDictionary(unittest.TestCase):
    def test_N(self):
        # Test that an error is raised if N specified in the parameters dictionary does not match the image dimensions
        params_dict = acq_params.copy()
        params_dict['N'] = 1
        with self.assertRaises(ValueError): ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=params_dict)
        with self.assertRaises(ValueError): ORC.fs_CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, Lx=1, nonCart=1, params=params_dict)
        with self.assertRaises(ValueError): ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, Lx=1, nonCart=1, params=params_dict)

    def test_Npoints(self):
        params_dict = acq_params.copy()
        params_dict['Npoints'] = 10
        with self.assertRaises(ValueError): ORC.CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, nonCart=1, params=params_dict)
        with self.assertRaises(ValueError): ORC.fs_CPR(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, Lx=1, nonCart=1, params=params_dict)
        with self.assertRaises(ValueError): ORC.MFI(dataIn=raw_data[:, :, 0], dataInType='raw', kt=ktraj,
                                                       df=field_map_CPR, Lx=1, nonCart=1, params=params_dict)





if __name__ == "__main__":
    unittest.main()