'''
Author: Marina Manso Jimeno
Last modified: 03/02/2020
'''

import scipy.io as sio
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math

from rawdata_recon import b0map_recon
from rawdata_recon import spiral_recon
from read_dicom import read_dicom_fieldmap

import ORC

##
# Load the input data (raw data, k-space trajectory and field map
##
data_path = '../../Data/20200624/shim/'

rawdata = sio.loadmat(data_path + 'rawdata_spiral')['dat']
ktraj = np.load(data_path + 'ktraj.npy')
dcf = sio.loadmat(data_path + 'dcf')['dcf_out'].flatten()
b0map = np.fliplr(nib.load(data_path + 'fieldmap_radspers.nii.gz').get_fdata()/(2 * math.pi))

if len(b0map.shape)==2:
    b0map = np.expand_dims(b0map, axis=2)

plt.imshow(np.rot90(b0map[:,:,0],-1), cmap='gray')
plt.colorbar()
plt.title('Field map')
plt.show()

#b0map = b0map_recon(data_path, method='HP',plot = 1)

##
# Dimensions check
##
if b0map.shape[0] != b0map.shape[1]:
    raise ValueError('Image and field map should have square dimensions (NxN)')
if rawdata.shape[0] != ktraj.shape[0] or rawdata.shape[1] != ktraj.shape[1]:
    raise ValueError('The raw data does not agree with the k-space trajectory')

##
# Useful parameters
##

FOV = 384e-3 # meters
N = b0map.shape[0] # Matrix size
Npoints = rawdata.shape[0]
Nshots = rawdata.shape[1]
Nchannels = rawdata.shape[-1]
if len(rawdata.shape) < 4:
    rawdata = rawdata.reshape(Npoints, Nshots, 1, Nchannels)
Nslices = rawdata.shape[2]
# Scaling factor (pyNUFFT) [-pi, pi]
ktraj_sc = math.pi / abs(np.max(ktraj))
ktraj = ktraj * ktraj_sc

t_ro = Npoints * 10e-6 # read-out time, hard-coded for Siemens-Pulseq
T = np.linspace(4.6e-3, 4.6e-3 + t_ro, Npoints).reshape((Npoints, 1))
seq_params = {'FOV': FOV, 'N': N, 'Npoints': Npoints, 'Nshots': Nshots, 'dcf': dcf, 't_vector': T, 't_readout': t_ro}

##
# Plot the original data
##
spiral_recon(data_path, ktraj, N, plot=1)

##
# Off resonance correction
##
Lx = 2
if Lx < 1:
    raise ValueError('The L factor cannot be lower that 1 (minimum L)')
CPR_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
fsCPR_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
MFI_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
for ch in range(Nchannels):
    for sl in range(Nslices):
        CPR_result[:, :, sl, ch] = ORC.CPR(np.squeeze(rawdata[:, :, sl, ch]), 'raw', ktraj,
                                                np.squeeze(b0map[:, :, sl]), 1, seq_params)
        print('fsCPR: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        fsCPR_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(rawdata[:, :, sl, ch]),'raw', ktraj, np.squeeze(b0map[:, :, sl]), Lx, 1, seq_params)
        print('fsCPR: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(data_path + 'fsCPR_Lx' + str(Lx), fsCPR_result)
        MFI_result[:, :, sl, ch] = ORC.MFI(np.squeeze(rawdata[:, :, sl, ch]),'raw', ktraj, np.squeeze(b0map[:, :, sl]), Lx, 1,seq_params)
        print('MFI: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(data_path + 'MFI_Lx' + str(Lx), MFI_result)

##
# Display the results
##
sos_CPR = np.sum(np.abs(CPR_result), 3)
sos_CPR = np.divide(sos_CPR, np.max(sos_CPR))

plt.imshow(np.rot90(sos_CPR[:,:,0],-1), cmap='gray')
plt.axis('off')
plt.title('CPR Corrected Image')
plt.show()

sos_fsCPR = np.sum(np.abs(fsCPR_result), 3)
sos_fsCPR = np.divide(sos_fsCPR, np.max(sos_fsCPR))

plt.imshow(np.rot90(sos_fsCPR[:,:,0],-1), cmap='gray')
plt.axis('off')
plt.title('fsCPR Corrected Image')
plt.show()

sos_MFI = np.sum(np.abs(MFI_result), 3)
sos_MFI = np.divide(sos_MFI, np.max(sos_MFI))

plt.imshow(np.rot90(sos_MFI[:,:,0],-1), cmap='gray')
plt.axis('off')
plt.title('MFI Corrected Image')
plt.show()

