"""
Small example / smoke test for the xtal_light_up Rust module.

Runs each of the module's functions on a single element and, for the in-fiber
check, also computes the distance to each fiber with HEXRD so the two can be
compared. Requires the xtal_light_up module to be built
(``pip install .`` from this directory) and HEXRD to be installed.

For processing actual ExaConstit results see light_up_py/fiber_calcs_rank.py.
"""
import numpy as np
import xtal_light_up.xtal_light_up as xlup

from hexrd import rotations as rot
from hexrd import material , valunits

from hexrd.matrixutil import  \
    columnNorm, unitVector, \
    skewMatrixOfVector, findDuplicateVectors, \
    multMatArray, nullSpace

def make_matl(mat_name , sgnum , lparms , hkl_ssq_max = 50 , dmin_angstroms = 0.5) :
    """
    Build a HEXRD material, used here for its plane data (reciprocal lattice B and
    crystal symmetry) when comparing against HEXRD.

    Parameters
    ----------
    mat_name : str
        label for material.
    sgnum : int
        space group number for material.
    lparms : list of floats
        lattice parameters in angstroms.
    hkl_ssq_max : int, optional
        maximum hkl sum of squares (peak upper bound). The default is 50.
    dmin_angstroms : float, optional
        minimum d-spacing in angstroms (alt peak upper bound). The default is 0.5.

    Returns
    -------
    hexrd.material.Material
    """
    
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max
    matl.dmin = valunits.valWUnit('lp' , 'length' , dmin_angstroms , 'angstrom')
    
    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls , dtype = bool))
    
    return matl

def applySym(vec, qsym, csFlag=False, cullPM=False, tol=rot.cnst.sqrt_epsf):
    """
    Apply a symmetry group to a single 3-vector (columnar) argument, adapted from
    HEXRD's version.

    Parameters
    ----------
    vec : (3, 1) array
        Vector to apply the symmetry operations to.
    qsym : (4, nsym) array
        Symmetry operations as quaternions (e.g. HEXRD's PlaneData.getQSym()).
    csFlag : bool, optional
        Centrosymmetry flag: also include -1 * each symmetric vector.
    cullPM : bool, optional
        Treat +v and -v as duplicates when removing duplicate vectors.
    tol : float, optional
        Tolerance for considering two vectors duplicates.

    Returns
    -------
    (3, m) array
        The m unique symmetrically equivalent vectors.
    """
    nsym = qsym.shape[1]
    Rsym = rot.rotMatOfQuat(qsym)
    if nsym == 1:
        Rsym = np.array([Rsym, ])
    allhkl = multMatArray(
        Rsym, np.tile(vec, (nsym, 1, 1))
    ).swapaxes(1, 2).reshape(nsym, 3).T

    if csFlag:
        allhkl = np.hstack([allhkl, -1*allhkl])
    eqv, uid = findDuplicateVectors(allhkl, tol=tol, equivPM=cullPM)

    return allhkl[np.ix_(list(range(3)), uid)]

def distanceToFiber(c, s, q, qsym, **kwargs):
    """
    Calculate the symmetrically reduced distance (angle) of each orientation to
    the fiber defined by crystal direction c and sample direction s, adapted from
    HEXRD's version. This is the pure Python / HEXRD equivalent of what
    xlup.calc_within_fibers computes before applying its tolerance, so the two
    can be compared.

    Parameters
    ----------
    c : 3-vector
        Crystal direction (HKL components, converted with ``bmatrix`` if given).
    s : 3-vector
        Sample direction.
    q : (4, nq) array
        Orientations as quaternions.
    qsym : (4, nsym) array
        Crystal symmetry operations as quaternions.
    **kwargs
        ``centrosymmetry`` (bool, default False): also consider -c.
        ``bmatrix`` ((3, 3) array, default identity): matrix taking c to the
        crystal frame (e.g. the reciprocal lattice B for HKL plane normals).

    Raises
    ------
    RuntimeError
        If c or s aren't 3-vectors or an unknown keyword argument is given.

    Returns
    -------
    d : (nq,) array
        Angle in radians between s and the closest symmetrically equivalent
        rotated crystal direction for each orientation.
    """
    csymFlag = False
    B = np.eye(3)

    arglen = len(kwargs)

    if len(c) != 3 or len(s) != 3:
        raise RuntimeError('c and/or s are not 3-vectors')

    # argument handling
    if arglen > 0:
        argkeys = list(kwargs.keys())
        for i in range(arglen):
            if argkeys[i] == 'centrosymmetry':
                csymFlag = kwargs[argkeys[i]]
            elif argkeys[i] == 'bmatrix':
                B = kwargs[argkeys[i]]
            else:
                raise RuntimeError("keyword arg \'%s\' is not recognized"
                                   % (argkeys[i]))

    c = unitVector(np.dot(B, np.asarray(c)))
    s = unitVector(np.asarray(s).reshape(3, 1))

    nq = q.shape[1]  # number of quaternions
    rmats = rot.rotMatOfQuat(q)  # (nq, 3, 3)

    csym = applySym(c, qsym, csymFlag)  # (3, m)
    m = csym.shape[1]  # multiplicity

    if nq == 1:
        rc = np.dot(rmats, csym)  # apply q's to c's
        sdotrc = np.dot(s.T, rc).max()
    else:
        rc = multMatArray(
            rmats, np.tile(csym, (nq, 1, 1))
        )  # apply q's to c's

        sdotrc = np.dot(
            s.T,
            rc.swapaxes(1, 2).reshape(nq*m, 3).T
        ).reshape(nq, m).max(1)

    d = rot.arccosSafe(np.array(sdotrc))

    print(d)

    return d


# help(xlup)

# A single element at a single step: arrays are (steps, elements, ...) just like the
# data read from ExaConstit's output in light_up_py/fiber_calcs_rank.py.
# Orientation quaternion [w, x, y, z] (normalized) and a lattice frame strain
# 6-vector [11, 22, 33, 23, 13, 12].
lattice_orientations = np.zeros((1,1,4))
strains = np.zeros((1,1,6))

lattice_orientations[:,:,0] = 1.0 / np.sqrt(1.5)
lattice_orientations[:,:,1] = 0.5 / np.sqrt(1.5)
lattice_orientations[:,:,2] = 0.5 / np.sqrt(1.5)

strains[:,:,0] = 1.0
strains[:,:,1] = 2.0
strains[:,:,2] = 3.0
strains[:,:,3] = 1.0
strains[:,:,4] = 2.0
strains[:,:,5] = 3.0

# Rotate the strain into the sample frame (in place). ExaConstit v0.9+ already
# writes its "Elastic Strains" in the sample frame, so this is only for older data.
xlup.strain_lattice2sample(lattice_orientations, strains)

# Taylor factor = sum of |slip system shear rates| / equivalent plastic strain rate.
# Both come straight from ExaConstit's outputs ("Shearing Rate" and ExaCMech's
# "Equivalent Plastic Strain Rate"), so nothing needs to be recomputed here. The values
# below just stand in for a single element's worth of that output; any number of slip
# systems works.
gdots = np.zeros((1,1,12))
gdots[:,:,0] = 1e-4
gdots[:,:,1] = 1e-4
gdots[:,:,2] = -1e-4
eq_pl_strain_rate = np.full((1,1), 1.0e-4)
tay_factor = np.zeros((1,1))

xlup.calc_taylor_factors(gdots, eq_pl_strain_rate, tay_factor)

# In-fiber check: compare HEXRD's distance to each HKL fiber with the Rust version.
# Get material symmetry - here is a simple example for IN625 (FCC, a = 3.60 angstrom)
matl = make_matl(mat_name = 'FCC' , sgnum = 225 , lparms = [3.60 ,])
pd = matl.planeData

hkl = np.zeros((4, 3))
hkl[0, :] = [1,1,1]
hkl[1, :] = [2,0,0]
hkl[2, :] = [2,2,0]
hkl[3, :] = [3,1,1]
s_dir = np.asarray([0.0,0.0,1.0])
quats = np.swapaxes(lattice_orientations, 0, 2)
for jj in range(4):
    # compute crystal direction from planeData
    c_dir = np.atleast_2d(np.dot(pd.latVecOps['B'] , hkl[jj, :].T)).T
    distance = distanceToFiber(c_dir, s_dir, quats, pd.getQSym())

# (nhkl, steps, elements) flags; lattice_type / lattice_params use the same names
# and parameters as ExaConstit's laue_type / lattice_params light-up options
in_fibers = np.zeros((hkl.shape[0], 1, 1), dtype=bool)

xlup.calc_within_fibers(lattice_orientations, s_dir, hkl, [3.60], np.deg2rad(5.0), in_fibers, lattice_type="cubic")

# # compute distance from quaternions to crystallographic fiber
# distance = np.degrees(rot.distanceToFiber(c_dir , s_dir , quats , pd.getQSym()))

# # filter for distances within specified distance bound
# in_fiber = np.where(distance < distance_bnd)[0]
# in_fiber_arr[jj] = in_fiber
