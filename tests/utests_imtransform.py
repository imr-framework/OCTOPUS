# Copyright of the Board of Trustees of Columbia University in the City of New York
import unittest
import matplotlib.pyplot as plt
import numpy as np

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from OCTOPUS.recon.imtransforms import im2ksp, ksp2im, nufft_init

# Cartesian data
N = 192
ph = resize(shepp_logan_phantom(), (N,N)).astype(complex)
cart_ksp = np.fft.fftshift(np.fft.fft2(ph))

# Non-Cartesian data
ktraj = np.load('test_data/ktraj_noncart.npy')
ktraj_dcf = np.load('test_data/ktraj_noncart_dcf.npy').flatten()

noncart_ksp = np.load('test_data/slph_noncart_ksp.npy')
acq_params = {'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 'N': ph.shape[0],'dcf': ktraj_dcf}


class TestImageTransformation(unittest.TestCase):
    def test_im2ksp_cart(self):
        # Image to k-space transformation for Cartesian data
        ksp = im2ksp(M=ph, cartesian_opt=1)
        self.assertEqual(ksp.shape, ph.shape) # Dimensions agree
        self.assertEqual(((ksp - cart_ksp) == np.zeros(ksp.shape).astype(complex)).all(), True) # Transformation is correct

    def test_ksp2im_cart(self):
        # K-space  to image transformation for Cartesian data
        im = ksp2im(ksp=cart_ksp, cartesian_opt=1)
        self.assertEqual(im.shape, cart_ksp.shape)  # Dimensions agree
        self.assertEqual((np.round(im - ph) == np.zeros(im.shape).astype(complex)).all(), True)  # Transformation is correct

    def test_im2ksp_noncart(self):
        # Image to k-space transformation for non-Cartesian data
        nufft_obj = nufft_init(kt=ktraj, params=acq_params)
        ksp = im2ksp(M=ph, cartesian_opt=0, NufftObj=nufft_obj, params=acq_params) # Sample the image along the trajectory
        self.assertEqual(ksp.shape, ktraj.shape) # Dimensions match

    def test_ksp2im_noncart(self):
        # K-space to image transformation for non-Cartesian
        nufft_obj = nufft_init(kt=ktraj, params=acq_params)
        im = ksp2im(ksp=noncart_ksp, cartesian_opt=0, NufftObj=nufft_obj, params=acq_params) # reconstructed image
        self.assertEqual(im.shape, ph.shape) # dimensions match
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) # normalize image
        diff = np.round(im - ph)
        diff_percent = np.nonzero(diff)[0].shape[0] / N**2 * 100
        self.assertLess(diff_percent, 5) # Less than 5% of voxels should be different between original and reconstructed image

class TestRaiseErrors(unittest.TestCase):
    # Test that error messages are raised when given invalid inputs
    def test_cartesian_opt(self):
        # cartesian_opt can only be 0 or 1, otherwise error
        with self.assertRaises(ValueError): im2ksp(M=ph, cartesian_opt=5)
        with self.assertRaises(ValueError): ksp2im(ksp=cart_ksp, cartesian_opt=100)

    def test_missing_params_Nshots(self):
        # NUFFT for non-cartesian data will fail if a required parameter is missing from the paramters dictionary (e.g. Nshots)
        params = acq_params.copy()
        del params['Nshots']
        with self.assertRaises(ValueError): nufft_init(kt=ktraj, params=params )

    def test_missing_params_Npoints(self):
        # NUFFT for non-cartesian data will fail if a required parameter is missing from the paramters dictionary (e.g. Npoints)
        params = acq_params.copy()
        del params['Npoints']
        with self.assertRaises(ValueError): nufft_init(kt=ktraj, params=params)

    def test_missing_params_N(self):
        # NUFFT for non-cartesian data will fail if a required parameter is missing from the paramters dictionary (e.g. N)
        params = acq_params.copy()
        del params['N']
        with self.assertRaises(ValueError): nufft_init(kt=ktraj, params=params)


if __name__ == "__main__":
    unittest.main()