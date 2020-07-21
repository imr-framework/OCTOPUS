# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Main script for off-resonance correction
Author: Marina Manso Jimeno
Last modified: 07/18/2020
'''
import configparser
import scipy.io as sio
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math
import time
import os

from OCTOPUS.utils.get_data_from_file import get_data_from_file
from OCTOPUS.Recon.rawdata_recon import fmap_recon
from OCTOPUS.Recon.rawdata_recon import spiral_recon
from OCTOPUS.Recon.read_dicom import read_dicom
from OCTOPUS.utils.plot_results import plot_correction_results

import OCTOPUS.ORC as ORC

# Read settings.ini configuration file
path_settings = 'settings.ini'
config = configparser.ConfigParser()
config.read(path_settings)
inputs = config['CORRECTION INPUTS']
outputs = config['CORRECTION OUTPUTS']

##
# Load the input data (raw data, k-space trajectory and field map)
##
FOV = 384e-3 # meters
dt = 10e-6 # seconds
TE = 4.6e-3 # seconds

rawdata = get_data_from_file(inputs['path_rawdatspiral_file'])
ktraj = get_data_from_file(inputs['path_ktraj_file'])

# Optional
if inputs['path_dcf_file']:
    dcf = get_data_from_file(inputs['path_dcf_file']).flatten() # Density correction factor

if inputs['path_fmapunwrapped_file']:
    fmap = np.fliplr(get_data_from_file(inputs['path_fmapunwrapped_file']) / (2 * math.pi))
else:
    fmap = fmap_recon(inputs['path_rawfmap_file'], 2.46e-3,  method='HP', plot = 0, save = 0)

if len(fmap.shape) == 2:
    fmap = np.expand_dims(fmap, axis=2)

plt.imshow(np.rot90(fmap[:,:,0], -1), cmap='gray')
plt.axis('off')
plt.colorbar()
plt.title('Field map')
plt.show()

##
# Dimensions check
##

if fmap.shape[0] != fmap.shape[1]:
    raise ValueError('Image and field map should have square dimensions (NxN)')
if rawdata.shape[0] != ktraj.shape[0] or rawdata.shape[1] != ktraj.shape[1]:
    raise ValueError('The raw data does not agree with the k-space trajectory')

##
# Useful parameters
##
N = fmap.shape[0] # Matrix size
Npoints = rawdata.shape[0]
Nshots = rawdata.shape[1]
Nchannels = rawdata.shape[-1]
if len(rawdata.shape) < 4:
    rawdata = rawdata.reshape(Npoints, Nshots, 1, Nchannels)
Nslices = rawdata.shape[2]

t_ro = Npoints * dt # read-out time, hard-coded for Siemens-Pulseq
T = np.linspace(TE, TE + t_ro, Npoints).reshape((Npoints, 1))
seq_params = {'FOV': FOV, 'N': N, 'Npoints': Npoints, 'Nshots': Nshots, 't_vector': T, 't_readout': t_ro}
#if 'dcf' in globals():
#    seq_params.update({'dcf': dcf})


##
# Plot the original data
##
original_im = spiral_recon(inputs['path_rawdatspiral_file'], ktraj, N, plot=1, save=0)

##
# Off resonance correction
##
if not os.path.isdir(outputs['path_correction_folder']):
    os.mkdir(outputs['path_correction_folder'])

Lx = 2 # Frequency segments for fs-CPR and MFI. L = Lx * Lmin
if Lx < 1:
    raise ValueError('The L factor cannot be lower that 1 (minimum L)')

CPR_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
fsCPR_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
MFI_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
CPR_timing = 0
fsCPR_timing = 0
MFI_timing = 0
for ch in range(Nchannels):
    for sl in range(Nslices):
        before = time.time()
        CPR_result[:, :, sl, ch] = ORC.CPR(np.squeeze(rawdata[:, :, sl, ch]), 'raw', ktraj,
                                                np.squeeze(fmap[:, :, sl]), 1, seq_params)
        CPR_timing += time.time() - before

        print('CPR: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(outputs['path_correction_folder'] + 'CPR', CPR_result)

        before = time.time()
        fsCPR_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(rawdata[:, :, sl, ch]),'raw', ktraj, np.squeeze(fmap[:, :, sl]), Lx, 1, seq_params)
        fsCPR_timing += time.time() - before

        print('fsCPR: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(outputs['path_correction_folder'] + 'fsCPR_Lx' + str(Lx), fsCPR_result)

        before = time.time()
        MFI_result[:, :, sl, ch] = ORC.MFI(np.squeeze(rawdata[:, :, sl, ch]),'raw', ktraj, np.squeeze(fmap[:, :, sl]), Lx, 1,seq_params)
        MFI_timing += time.time() - before

        print('MFI: Done with slice:' + str(sl + 1) + ', channel:' + str(ch + 1))
        np.save(outputs['path_correction_folder'] + 'MFI_Lx' + str(Lx), MFI_result)

##
# Display the results
##
print('\nCPR correction took ' + str(CPR_timing) + ' seconds.')
print('\nFs-CPR correction took ' + str(fsCPR_timing) + ' seconds.')
print('\nMFI correction took ' + str(MFI_timing) + ' seconds.')

sos_CPR = np.sum(np.abs(CPR_result), -1)
sos_CPR = np.divide(sos_CPR, np.max(sos_CPR))

sos_fsCPR = np.sum(np.abs(fsCPR_result), -1)
sos_fsCPR = np.divide(sos_fsCPR, np.max(sos_fsCPR))

sos_MFI = np.sum(np.abs(MFI_result), -1)
sos_MFI = np.divide(sos_MFI, np.max(sos_MFI))

im_stack = np.stack((np.squeeze(np.rot90(original_im,-1)), np.squeeze(np.rot90(sos_CPR,-1)), np.squeeze(np.rot90(sos_fsCPR,-1)), np.squeeze(np.rot90(sos_MFI,-1))))
cols = ('Corrupted Image','CPR Correction', 'fs-CPR Correction', 'MFI Correction')
row_names = (' ', )
plot_correction_results(im_stack, cols, row_names)



