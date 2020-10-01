import argparse
import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0,r'C:\Users\marin\Documents\GitHubRepositories\OCTOPUS')

from OCTOPUS.utils.dataio import get_data_from_file
import OCTOPUS.ORC as ORC


parser = argparse.ArgumentParser()
parser.add_argument("path", help="echo the string you use here")
parser.add_argument("datain", help="echo the string you use here")
parser.add_argument("ktraj", help="echo the string you use here")
parser.add_argument("fmap", help="echo the string you use here")
parser.add_argument("ORCmethod", choices=['CPR', 'fsCPR', 'MFI'], help="echo the string you use here")
parser.add_argument('--Lx', default=2, dest='Lx')
parser.add_argument('--grad_raster', default=10e-6, dest='grad_raster')
parser.add_argument('--TE', default=0, dest='TE')

#li = [r'C:\Users\marin\Documents\PhD\B0inhomogeneity\Data\20200917\shimmed', 'rawdata_spiral.mat', 'ktraj.npy', 'fieldmap_unwrapped.nii.gz', 'MFI']


#args = parser.parse_args(li)
args = parser.parse_args()

data_in = get_data_from_file(os.path.join(args.path, args.datain))
ktraj = get_data_from_file(os.path.join(args.path, args.ktraj))
fmap = get_data_from_file(os.path.join(args.path, args.fmap))

## Dimensions check
print('Checking dimensions...')
if data_in.shape[0] != ktraj.shape[0] or data_in.shape[1] != ktraj.shape[1]:
    raise ValueError('The raw data does not agree with the k-space trajectory')
if fmap.shape[0] != fmap.shape[1]:
    raise ValueError('Image and field map should have square dimensions (NxN)')
if len(fmap.shape) > 2:
    if fmap.shape[-1] != data_in.shape[2]:
        raise ValueError('The field map dimensions do not agree with the raw data')
if len(fmap.shape) == 2:
    fmap = np.expand_dims(fmap,-1)
    if data_in.shape[2] != 1:
        raise ValueError('The field map dimensions do not agree with the raw data')

print('OK')
'''print(data_in.shape)
print(ktraj.shape)
print(fmap.shape)
print(args.Lx)
print(args.grad_raster)'''

## Sequence parameters
N = fmap.shape[0]
Npoints = data_in.shape[0]
Nshots = data_in.shape[1]
Nslices = data_in.shape[-2]
Nchannels = data_in.shape[-1]

t_ro = Npoints * args.grad_raster
T = np.linspace(args.TE, args.TE + t_ro, Npoints).reshape((Npoints, 1))
seq_params = {'N': N, 'Npoints': Npoints, 'Nshots': Nshots, 't_vector': T, 't_readout': t_ro}

ORC_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)
if args.ORCmethod == 'MFI':
    for ch in tqdm(range(Nchannels)):
        for sl in range(Nslices):
            ORC_result[:, :, sl, ch] = ORC.MFI(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj, np.squeeze(fmap[:, :, sl]),
                                               args.Lx, 1, seq_params)
            print(ch)
elif args.ORCmethod == 'fsCPR':
    for ch in range(Nchannels):
        for sl in range(Nslices):
            ORC_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj, np.squeeze(fmap[:, :, sl]),
                                               args.Lx, 1, seq_params)

elif args.ORCmethod == 'CPR':
    for ch in tqdm(range(Nchannels)):
        for sl in range(Nslices):
            ORC_result[:, :, sl, ch] = ORC.CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                       np.squeeze(fmap[:, :, sl]), 1, seq_params)


