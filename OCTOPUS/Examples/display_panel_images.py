# Copyright of the Board of Trustees of Columbia University in the City of New York
'''
Displays a panel with column: Gold standard, Uncorrected spiral, Spiral CPR corrected, spiral fs-CPR corrected and spiral MFI corrected
Author: Marina Manso Jimeno
Last updated: 07/08/2020
'''
import numpy as np
import configparser
from OCTOPUS.utils.dataio import read_dicom
from OCTOPUS.utils.plotting import plot_correction_results
from OCTOPUS.utils.metrics import create_table


# Read settings.ini configuration file
path_settings = 'settings.ini'
config = configparser.ConfigParser()
config.read(path_settings)
paths = list(config['DISPLAY'].items())

data_paths = [path[1] for path in paths if path[1] != '']

N = 192
Nranges = len(data_paths)


GS = []
uncorrected = []
CPR = []
fs_CPR = []
MFI = []
for path in data_paths:
    vol_GS = read_dicom(path + 'GRE.dcm')
    vol_uncorr = np.load(path + 'spiral_bc_im.npy')
    vol_CPR = np.load(path + 'corrections/CPR.npy')
    vol_fsCPR = np.load(path + 'corrections/fsCPR_Lx2.npy')
    vol_MFI = np.load(path + 'corrections/MFI_Lx2.npy')


    sos_CPR = np.divide(np.sqrt(np.sum(np.abs(vol_CPR)**2, -1)), np.max(np.sqrt(np.sum(np.abs(vol_CPR)**2, -1))))
    sos_fsCPR = np.divide(np.sqrt(np.sum(np.abs(vol_fsCPR)**2, -1)), np.max(np.sqrt(np.sum(np.abs(vol_fsCPR)**2, -1))))
    sos_MFI = np.divide(np.sqrt(np.sum(np.abs(vol_MFI)**2, -1)), np.max(np.sqrt(np.sum(np.abs(vol_MFI)**2, -1))))

    GS.append(np.rot90(vol_GS,0))
    uncorrected.append(np.rot90(vol_uncorr,-1))
    CPR.append(np.rot90(sos_CPR, -1))
    fs_CPR.append(np.rot90(sos_fsCPR, -1))
    MFI.append(np.rot90(sos_MFI,-1))

GS = np.fliplr(np.moveaxis(np.squeeze(np.stack(GS)), 0, -1))
uncorrected = np.fliplr(np.moveaxis(np.stack(np.squeeze(uncorrected)), 0, -1))
CPR = np.fliplr(np.moveaxis(np.stack(np.squeeze(CPR)), 0, -1))
fs_CPR = np.fliplr(np.moveaxis(np.stack(np.squeeze(fs_CPR)), 0, -1))
MFI = np.fliplr(np.moveaxis(np.stack(np.squeeze(MFI)), 0, -1))

cols = ('Gold Standard','Corrupted Image','CPR Correction', 'fs-CPR Correction', 'MFI Correction')
#rows = ('Shimmed', 'Mid-shimmed', 'Not shimmed')
rows = ('Shimmed', 'X shim = -90', 'X shim = 0')
im_stack = np.stack((GS, uncorrected, CPR, fs_CPR, MFI))

plot_correction_results(im_stack, cols, rows)

#cols = ('Gold Standard', 'Corrupted Image','CPR Correction', 'fs-CPR Correction', 'MFI Correction')
#create_table(im_stack, cols, rows)
