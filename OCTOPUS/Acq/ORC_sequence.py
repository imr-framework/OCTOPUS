# Copyright of the Board of Trustees of Columbia University in the City of New York

'''
Creates the spiral .seq file to run on the scanner.
Author: Marina Manso Jimeno
Last modified: 02/26/2020
'''
##
# Imports
##
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.utils.vds_2d import vds_2d
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_adc import make_adc
from pypulseq.calc_duration import calc_duration
from pypulseq.make_delay import make_delay
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

from datetime import date

##
# Sequence parameters
##
# Siemens Pulseq
gamma = 42576000 # in Hz  (Determined from Pulseq - do not change)
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', grad_raster_time=10e-6)
seq = Sequence(system)

# Acquisition parameters
FOV = 384e-3 # meters
N = 192 # Matrix size
Nslices = 1
sl_thickness = 3e-3 # meters
sl_gap = 0
Ndummy = 5

FA = 30 # degrees
TR = 25e-3 # seconds
TE = 4.6e-3 #2.2e-3

# Spiral parameters
Nshots = 54 # Number of spiral arms
alpha = 9 # center oversampling factor
ktraj, G, _ = vds_2d(FOV, N, Nshots, alpha, system)

##
# Sequence blocks
##

# RF & gradient for Slice selection
rf,  gz, _ = make_sinc_pulse(flip_angle=FA * math.pi/180, duration=3e-3, slice_thickness=sl_thickness, time_bw_product=4, system=system)
dz = (Nslices - 1) * (sl_thickness + sl_gap)
step = sl_thickness + sl_gap
z = np.arange(-dz / 2, dz/2 + step/2, step)

# Spoiler and rephasing gradients
gzReph = make_trapezoid(channel='z', system=system, area=-gz.area/2, duration=3e-3)
gzSpoil = make_trapezoid(channel='z', system=system, area=gz.area*2, duration=8*8e-4)

# adc
adc = make_adc(num_samples=len(G), dwell=system.grad_raster_time, system=system)

##
# Sequence Timing
##
delay1 = TR - calc_duration(rf) / 2 - calc_duration(gzReph) # Dummy pulses
delay_dummy = make_delay(delay1)
delay2 = TE - calc_duration(rf)/2 - calc_duration(gzReph) # TE
delay_TE = make_delay(delay2)

##
# Sequence Blocks
##
for sl in range(Nslices):
    freq_offset = gz.amplitude * z[sl]
    rf.freqOffset = freq_offset

    # Dummy Pulses
    for nrf in range(Ndummy):
        seq.add_block(rf,gz)
        seq.add_block(gzReph)
        seq.add_block(delay_dummy)

    for nshot in range(Nshots):
        # Gradient waveforms
        gx = make_arbitrary_grad(channel='x', waveform=np.squeeze(G[:, nshot].real), system=system)
        gy = make_arbitrary_grad(channel='y', waveform=np.squeeze(G[:, nshot].imag), system=system)

        seq.add_block(rf, gz)
        seq.add_block(gzReph)
        seq.add_block(delay_TE)
        delay3 = TR - delay2 - calc_duration(gx) - calc_duration(gzSpoil)
        delay_TR = make_delay(delay3)

        seq.add_block(gx,gy,adc)
        seq.add_block(gzSpoil)
        seq.add_block(delay_TR)


seq.plot(time_range=(0.125, 9*TR))
plt.show()

##
# Save the files
##
seq_name = str(date.today()) + '_ORC_' +str(Nshots) + '_' + str(alpha) + '_' + str(int(FOV*1e3)) + '_' + str(Nslices)
seq.write('seq_' + seq_name)
np.save(str(date.today()) + '_ktraj', ktraj)
ktrajs = np.zeros((ktraj.shape[0], Nshots, 2))
ktrajs[:,:,0] = ktraj.real
ktrajs[:,:,1] = ktraj.imag
sio.savemat(str(date.today()) + '_ktraj.mat', {'ktrajs': ktrajs})

