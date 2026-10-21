#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shows how to plot stress strain curves from the surface quantity files and compare them
with the volume average stress file

Needs [PostProcessing.surface_quantities] with the loaded surface and the opposite
surface in boundary_ids (for an auto generated mesh loaded along z: 4 = top, 1 = bottom)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('mathtext', default='regular')

font = {'size'   : 14}

rc('font', **font)
# Change these to fit your simulation
fileLoc = 'path of dir of avg_stress_global.txt and surface_quantities_surface_*.txt/'
top_id = 4
bottom_id = 1
# Loading direction: 0 = x, 1 = y, 2 = z
axis = 2
# Length of the sample along the loading direction
length = 1.0

# Columns of the surface files: time, surface ID, area, disp_x, disp_y, disp_z,
# force_x, force_y, force_z, normal_x, normal_y, normal_z
# uncomment the below when the fileLoc is valid
#top = np.loadtxt(fileLoc + 'surface_quantities_surface_' + str(top_id) + '.txt', comments='#')
#bottom = np.loadtxt(fileLoc + 'surface_quantities_surface_' + str(bottom_id) + '.txt', comments='#')
#vol_avg = np.loadtxt(fileLoc + 'avg_stress_global.txt', comments='#')
# only here to have something that'll plot
nsteps = 20
top = np.ones((nsteps, 12))
bottom = np.ones((nsteps, 12))
vol_avg = np.ones((nsteps, 8))

# Engineering strain from the change in length between the two surfaces
eps = (top[:, 3 + axis] - bottom[:, 3 + axis]) / length

# True stress from the surface force and the current area. Multiplying by the normal
# component makes the stress positive in tension for either surface.
sig_top = top[:, 6 + axis] * top[:, 9 + axis] / top[:, 2]
sig_bottom = bottom[:, 6 + axis] * bottom[:, 9 + axis] / bottom[:, 2]
# The first two columns of the volume average file are time and volume, followed by
# the Cauchy stress in voigt notation
sig_vol = vol_avg[:, 2 + axis]

# Start the curves at zero
eps = np.r_[0, eps]
sig_top = np.r_[0, sig_top]
sig_bottom = np.r_[0, sig_bottom]
sig_vol = np.r_[0, sig_vol]

fig, ax = plt.subplots(1)

# For true strain the np.log(1 + eps) will provide the correct conversion
ax.plot(np.log(1.0 + eps), sig_vol, color='blue', linestyle='solid', label='Volume average')
ax.plot(np.log(1.0 + eps), sig_top, color='orange', linestyle='--', label='Top surface')
ax.plot(np.log(1.0 + eps), sig_bottom, color='green', linestyle=':', label='Bottom surface')
ax.grid()
ax.legend()

# change this to fit your data
# ax.axis([0, 0.01, 0, 0.3])

ax.set_ylabel('Macroscopic true stress [GPa]')
ax.set_xlabel('Macroscopic true strain [-]')

# Percent difference of the surface stresses from the volume average, skipping the zero
# starting point
diff_top = 100.0 * (sig_top[1:] - sig_vol[1:]) / np.abs(sig_vol[1:])
diff_bottom = 100.0 * (sig_bottom[1:] - sig_vol[1:]) / np.abs(sig_vol[1:])
print('Max difference from the volume average: top %.3f%%, bottom %.3f%%'
      % (np.max(np.abs(diff_top)), np.max(np.abs(diff_bottom))))

fig.show()
plt.show()
