import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import math

sys.path.insert(0,r'C:\Users\marin\Documents\GitHubRepositories\OCTOPUS')

from OCTOPUS.utils.dataio import get_data_from_file
import OCTOPUS.ORC as ORC


parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path containing the input data")
parser.add_argument("datain", help="Corrupted data with dimensions [Npoints OR N(Cartesian), Nshots OR N(Cartesian), Nslices, Nchannels]")
parser.add_argument("ktraj", help="K-space trajectory: [Npoints OR N(Cartesian), Nshots OR N(Cartesian)]")
parser.add_argument("fmap", help="Field map in rad/s")
parser.add_argument("ORCmethod", choices=['CPR', 'fsCPR', 'MFI'], help="Correction method")
parser.add_argument("--cart", default=0, choices=[0, 1], dest="cart_opt", help="0: non-cartesian data, 1: cartesian data")
parser.add_argument('--Lx', default=2, dest='Lx', help="L(frequency bins) factor with respect to minimum L")
parser.add_argument('--grad_raster', default=10e-6, dest='grad_raster', help="Gradient raster time")
parser.add_argument('--TE', default=0, dest='TE', help="Echo time")
parser.add_argument('--dcf', default=None, dest='dcf', help="Density compensation factor for non-cartesian trajectories")

#li = [r'C:\Users\marin\Documents\PhD\B0inhomogeneity\Data\20200917\shimmed', 'rawdata_spiral.mat', 'ktraj.npy', 'fieldmap_unwrapped.nii.gz', 'MFI']


#args = parser.parse_args(li)
args = parser.parse_args()

data_in = get_data_from_file(os.path.join(args.path, args.datain))
ktraj = get_data_from_file(os.path.join(args.path, args.ktraj))
fmap = get_data_from_file(os.path.join(args.path, args.fmap))/ (2 * math.pi)

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

## Sequence parameters
N = fmap.shape[0]
Npoints = data_in.shape[0]
Nshots = data_in.shape[1]
Nslices = data_in.shape[-2]
Nchannels = data_in.shape[-1]

t_ro = Npoints * args.grad_raster
T = np.linspace(args.TE, args.TE + t_ro, Npoints).reshape((Npoints, 1))
if args.cart == 0:
    seq_params = {'N': N, 'Npoints': Npoints, 'Nshots': Nshots, 't_vector': T, 't_readout': t_ro}

    if args.dcf is not None:
        dcf = get_data_from_file(os.path.join(args.path, args.dcf)).flatten()
        seq_params.update({'dcf': dcf})

ORC_result = np.zeros((N, N, Nslices, Nchannels), dtype=complex)


if args.ORCmethod == 'MFI':
    for ch in tqdm(range(Nchannels)):
        for sl in range(Nslices):
            if args.cart == 0:
                ORC_result[:, :, sl, ch] = ORC.MFI(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj, np.squeeze(fmap[:, :, sl]), args.Lx, 1, seq_params)
            elif args.cart == 1:
                ORC_result[:, :, sl, ch] = ORC.MFI(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                                   np.squeeze(fmap[:, :, sl]), args.Lx)

elif args.ORCmethod == 'fsCPR':
    for ch in tqdm(range(Nchannels)):
        for sl in range(Nslices):
            if args.cart == 0:
                ORC_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                                   np.squeeze(fmap[:, :, sl]), args.Lx, 1, seq_params)
            elif args.cart == 1:
                ORC_result[:, :, sl, ch] = ORC.fs_CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                                   np.squeeze(fmap[:, :, sl]), args.Lx)
elif args.ORCmethod == 'CPR':
    for ch in tqdm(range(Nchannels)):
        for sl in range(Nslices):
            if args.cart == 0:
                ORC_result[:, :, sl, ch] = ORC.CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                                   np.squeeze(fmap[:, :, sl]), 1, seq_params)
            elif args.cart == 1:
                ORC_result[:, :, sl, ch] = ORC.CPR(np.squeeze(data_in[:, :, sl, ch]), 'raw', ktraj,
                                                   np.squeeze(fmap[:, :, sl]))


np.save(os.path.join(args.path, 'ORC_result_' + args.ORCmethod), ORC_result)