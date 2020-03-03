'''
Author: Marina Manso Jimeno
Last modified: 03/02/2020
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import ORC
##
# Load the input data (raw data, k-space trajectory and field map
##
data_path = '../../Data/20200227/Shim/'

rawdata = sio.loadmat(data_path + 'rawdata_spiral')['dat']
ktraj = np.load(data_path + 'ktraj.npy')
dcf = np.ones(int(ktraj.shape[0] * ktraj.shape[1]))
b0map = np.fliplr(np.load(data_path + 'b0map.npy'))

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
# TODO: Select the L we want and include it as input
FOV = 384e-3 # meters
N = b0map.shape[0] # Matrix size
Npoints = rawdata.shape[0]
Nshots = rawdata.shape[1]
Nchannels = rawdata.shape[-1]
if len(rawdata.shape) < 4:
    rawdata = rawdata.reshape(Npoints, Nshots, 1, Nchannels)
Nslices = rawdata.shape[2]
ktraj = ktraj / 80 # Scaling factor (pyNUFFT)
t_ro = Npoints * 10e-6 # read-out time, hard-coded for Siemens-Pulseq
T = np.linspace(0, t_ro, Npoints).reshape((Npoints, 1))
seq_params = {'FOV': FOV, 'N': N, 'Npoints': Npoints, 'Nshots': Nshots, 'dcf': dcf, 't_vector': T, 't_readout': t_ro}

##
# Off resonance correction
##
fsCPR_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
MFI_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
for ch in range(Nchannels):
    for sl in range(Nslices):
        fsCPR_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(rawdata[:, :, sl, ch]), ktraj, np.squeeze(b0map[:, :, sl]), seq_params)
        print('fsCPR: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(data_path + 'fsCPR', fsCPR_result)
        MFI_result[:, :, sl, ch] = ORC.MFI(np.squeeze(rawdata[:, :, sl, ch]), ktraj, np.squeeze(b0map[:, :, sl]), seq_params)
        print('MFI: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(data_path + 'MFI', MFI_result)

##
# Display the results
##
sos_fsCPR = np.sum(np.abs(fsCPR_result), 3)
sos_fsCPR = np.divide(sos_fsCPR, np.max(sos_fsCPR))

plt.imshow(np.rot90(np.abs(sos_fsCPR[:,:,0]),-1), cmap='gray')
plt.title('fsCPR Corrected Image')
plt.show()

sos_MFI = np.sum(np.abs(MFI_result), 3)
sos_MFI = np.divide(sos_MFI, np.max(sos_MFI))

plt.imshow(np.rot90(np.abs(sos_MFI[:,:,0]),-1), cmap='gray')
plt.title('MFI Corrected Image')
plt.show()

