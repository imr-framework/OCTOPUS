'''
Displays a panel with column: Gold standard, Uncorrected spiral, Spiral CPR corrected, spiral fs-CPR corrected and spiral MFI corrected
Author: Marina Manso Jimeno
Last updated: 07/08/2020
'''
import numpy as np
from Recon.read_dicom import read_dicom_gre
from utils.plot_restults import plot_correction_results

data_paths = ['../../Data/20200708/shim/', '../../Data/20200708/mid_shim/', '../../Data/20200708/no_shim/']

N = 192
Nranges = len(data_paths)


GS = []
uncorrected = []
CPR = []
fs_CPR = []
MFI = []
for path in data_paths:
    vol_GS = read_dicom_gre(path + 'GRE_GS_dicom.IMA')
    vol_uncorr = np.load(path + 'uncorrected_spiral.npy')
    vol_CPR = np.load(path + 'corrections/CPR.npy')
    vol_fsCPR = np.load(path + 'corrections/fsCPR_Lx2.npy')
    vol_MFI = np.load(path + 'corrections/MFI_Lx2.npy')


    sos_CPR = np.divide(np.sum(np.abs(vol_CPR), -1), np.max(np.sum(np.abs(vol_CPR), -1)))
    sos_fsCPR = np.divide(np.sum(np.abs(vol_fsCPR), -1), np.max(np.sum(np.abs(vol_fsCPR), -1)))
    sos_MFI = np.divide(np.sum(np.abs(vol_MFI), -1), np.max(np.sum(np.abs(vol_MFI), -1)))

    GS.append(vol_GS)
    uncorrected.append(np.rot90(vol_uncorr,-1))
    CPR.append(np.rot90(sos_CPR, -1))
    fs_CPR.append(np.rot90(sos_fsCPR, -1))
    MFI.append(np.rot90(sos_MFI,-1))

GS = np.moveaxis(np.squeeze(np.stack(GS)), 0, -1)
uncorrected = np.moveaxis(np.stack(np.squeeze(uncorrected)), 0, -1)
CPR = np.moveaxis(np.stack(np.squeeze(CPR)), 0, -1)
fs_CPR = np.moveaxis(np.stack(np.squeeze(fs_CPR)), 0, -1)
MFI = np.moveaxis(np.stack(np.squeeze(MFI)), 0, -1)

cols = ('Gold Standard','Corrupted Image','CPR Correction', 'fs-CPR Correction', 'MFI Correction')
rows = ('Shimmed', 'Mid-shimmed', 'Not shimmed')
im_stack = np.stack((GS, uncorrected, CPR, fs_CPR, MFI))

plot_correction_results(im_stack, cols, rows)
