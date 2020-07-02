import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import ORC
import fieldmap_gen
from artifactID.common.data_ops import glob_brats_t1,load_nifti_vol
from artifactID.datagen import generate_fieldmap



path_brats = r'C:\Users\marin\Documents\PhD\Academic\Applied Deep Learning\Project\Data\MICCAI_BraTS_2018_Data_Training'
##
# Cartesian k-space trajectory
##
dt = 10e-6 # grad raster time
ktraj_cart = np.arange(0, 240 * dt, dt).reshape(1,240)
ktraj_cart = np.tile(ktraj_cart, (240, 1))

freq_range = 2500
data = glob_brats_t1(path_brats)
for ind, path_t1 in tqdm(enumerate(data)):
    vol = load_nifti_vol(path_t1)
    or_corrupted = np.zeros(vol.shape, dtype=complex)
    for ind in range(vol.shape[-1]):
        slice = vol[:, :, ind]
        fieldmap = fieldmap_gen.spherical_order4(240, freq_range)
        #fieldmap, mask = generate_fieldmap.gen_smooth_b0(slice, freq_range)
        or_corrupted[:, :, ind] = ORC.add_or_CPR(slice, ktraj_cart, fieldmap)
        if ind == 58:
            print('hello')