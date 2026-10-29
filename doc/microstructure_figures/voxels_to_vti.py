"""Writes the voxel map of an ExaConstit microstructure file as a VTK image (.vti) for ParaView.

  pvpython voxels_to_vti.py IN.h5 OUT.vti

The cell field grain_id holds the grain number of every voxel, so render_grains.py can draw the
data itself (no mesh or generator run needed). Uses h5py and VTK, both shipped with ParaView.
"""
import sys

import h5py
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

with h5py.File(sys.argv[1], "r") as h5:
    coords = h5["mesh/coordsets/coords"]
    values = np.asarray(h5["mesh/fields/grain_id/values"][()], dtype=np.int32)
    origin = [float(coords["origin/" + a][()]) for a in "xyz"]
    spacing = [float(coords["spacing/" + a][()]) for a in ("dx", "dy", "dz")]
nz, ny, nx = values.shape  # x varies fastest, as VTK expects for cell data

image = vtkImageData()
image.SetDimensions(nx + 1, ny + 1, nz + 1)
image.SetOrigin(origin)
image.SetSpacing(spacing)
array = numpy_to_vtk(values.ravel(), deep=True)
array.SetName("grain_id")
image.GetCellData().AddArray(array)

writer = vtkXMLImageDataWriter()
writer.SetFileName(sys.argv[2])
writer.SetInputData(image)
writer.Write()
print("wrote", sys.argv[2], "with", values.size, "voxels")
