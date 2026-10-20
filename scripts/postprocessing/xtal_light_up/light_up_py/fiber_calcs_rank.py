"""
Computes lattice strain / "light-up" quantities from an ExaConstit ADIOS2
visualization output (.bp) using the xtal_light_up Rust module.

For each HKL fiber (elements whose crystal orientation puts one of the HKL's
symmetrically equivalent plane normals within a tolerance of the sample
direction) and every output step it computes the volume averaged:

- lattice strain (sample frame elastic strain projected on the sample direction)
- directional stiffness (sample frame 33 Cauchy stress / 33 elastic strain)
- Taylor factor (sum of |slip system shear rates| / equivalent plastic strain rate)
- equivalent plastic strain rate
- in-fiber volume

These are the same quantities ExaConstit's in-situ light-up computes, just done
afterwards from the visualization output, for every crystal system ExaConstit
supports.

Requirements on the ExaConstit run (ExaCMech material model):

- [Visualizations] adios2 = true, which needs MFEM built with ADIOS2
  (MFEM_USE_ADIOS2). The .bp output is written to the visualization output
  location ([output_directory]/[basename]/[floc], see Visualizations.floc in
  options.toml); runs with output split by region write one .bp per region.
- [PostProcessing.projections] must include these fields, which is the default
  with auto_enable_compatible = true, or otherwise list them in
  enabled_projections:

  ===================  ===============================  ==========================
  enabled_projections  ADIOS2 variable read             used for
  ===================  ===============================  ==========================
  "volume"             Element Volumes                  volume weighting
  "xtal_ori"           Crystal Orientations             which elements are in fiber
  "elastic_strain"     Elastic Strains                  lattice strain, stiffness
  "cauchy"             Cauchy Stress                    directional stiffness
  "shear_rate"         Shearing Rate                    Taylor factor
  "dpeff"              Equivalent Plastic Strain Rate   Taylor factor, eq. pl. rate
  ===================  ===============================  ==========================

- The number of MPI ranks the simulation ran on (-nranks), since the ADIOS2
  output is written as one block per rank and every block has to be read.

Elastic strains are expected in the sample frame, which ExaConstit v0.9+ writes.
The directional stiffness, Taylor factor and equivalent plastic strain rate have
one fewer entry than the number of output steps, since the initial (undeformed)
output has no stress / rate data to go with it.

Usage:
    python fiber_calcs_rank.py -in_dir <path to the .bp output> \
        -out_dir ./outdir -out_basename rve_ -nranks 48 \
        -lattice_type hexagonal -lattice_params 2.95 4.68 -distance_tolerance_deg 5

The HKLs and sample direction are set near the top of fiber_calc_ranks. Results
are printed and saved to <out_dir>/<out_basename>lattice_avg_quants.npz with the
arrays lattice_strains, lattice_vols (nhkl, steps), lattice_dir_stiff,
lattice_tay_fact, lattice_eps_rate (nhkl, steps - 1), hkls and total_volume.
"""
# Python standard library imports
import argparse
import os
import time

# third-party library imports
import numpy as np
import adios2

#local imports
import xtal_light_up.xtal_light_up as xlup

def fiber_calc_ranks(args):
    """
    Reads the ExaConstit ADIOS2 output one rank's block at a time, accumulates the
    volume weighted light-up quantities for every HKL fiber, then normalizes them
    by the in-fiber volumes and prints / saves the results.

    Parameters
    ----------
    args : dict
        The command line options (see the module docstring / -h): in_dir,
        out_dir, out_basename, nranks, lattice_type, lattice_params and
        distance_tolerance_deg.
    """
    ts_total = time.time()
    # path to ExaConstit ADIOS2 binary-pack (.bp)
    in_dir = args["in_dir"]
    if not os.path.exists(in_dir) :
        raise Warning('Input directory not found, aborting!')

    # save directory for script outputs, recursively create directory if it doesn't exist
    out_dir = args["out_dir"]
    if not os.path.exists(out_dir) :
        print('Output directory not found, created at %s' % out_dir)
        os.makedirs(out_dir)
    else :
        print('Output directory found, proceeding.')

    # int - number of resource sets used for the simulation
    # IMPORTANT: must set this correctly to process all elements in the simulation
    nranks = args["nranks"] # 48

    # open ADIOS2 binary-pack (.bp) file
    ### fh = adios2.open(in_dir , 'r' , engine_type = 'BP4') # this doesn't work in my adios2 install - seems they've removed open
    fh = adios2.FileReader(in_dir)

    # list of variables stored in adios2 file
    init_vars = fh.available_variables()

    # total number of cycles saved off (+ initial step at time = 0)
    # if stride > 1 in options.toml, this number will be (# of ExaConstit steps / stride) + 1
    ### fh.steps() # doesn't work after switching to FileReader (see above)
    steps = fh.num_steps()

    #%% Extract connectivity information - needed for other variables. (NO INPUTS HERE)

    con1d = list()
    index = np.zeros((nranks , 2) , dtype = np.int32)
    iend = 0

    # Get the initial end node vertices and connectivity array for everything.
    # ADIOS2 doesn't save higher order values, so only have the end nodes.
    # Many repeat vertices, since these are saved off for each element.
    for i in range(nranks) :
        
        if (i == 0) :
            
            # Pull out connectivity information.
            con = fh.read('connectivity' , block_id = i)
        
            con1d.append(con[:,1])
            con = con[:,1::]
            
            # # Can uncomment to also pull out vertices.
            # vert = fh.read('vertices' , block_id = i)
            
            # # Can uncomment to also pull out grain IDs ('ElementAttribute').
            # grain = fh.read('ElementAttribute' , block_id = i)
            # grain = grain[con[:,1]]
            
        else :
            
            # Pull out connectivity information.
            tmp = fh.read('connectivity' , block_id = i)
            con1d.append(tmp[:,1])

            # Connectivity is local to resource set rather than global, so increment to global.
            tmp = tmp + np.max(con)
            con = np.vstack((con , tmp[:,1::]))
            
            # # Can uncomment to also pull out vertices.
            # tmp = fh.read('vertices' , block_id = i)
            # vert = np.vstack((vert , tmp))
            
            # # Can uncomment to also pull out grain IDs ('ElementAttribute').
            # tmp = fh.read('ElementAttribute' , block_id = i)
            # grain = np.hstack((grain , tmp[con1d[i]]))
            
            del tmp

        # indexing variable that will be used later on
        index[i,0] = iend
        iend = con.shape[0]
        index[i,1] = iend

    # # Can uncomment to convert grain IDs to int32.
    # grain = np.int32(grain)

    conshape = np.copy(con.shape)

    # list of variables to save off (can view available variables in init_vars in next block below)
    # different variables are stored in different ways - not all variables are supported by this script
    # this script should work for any variables that are saved off for every element - some examples of working variables are given below
    # vars_out = [
    #     'ElementVolume' ,
    #     'LatticeOrientation' ,
    #     'ShearRate' ,
    #     'Stress' ,
    #     'XtalElasticStrain'
    #     ]

    # initialize

    # HKL planes to compute the fibers for (Miller indices, 3-index form for
    # hexagonal / trigonal too) and the sample direction they're measured along
    hkl = np.zeros((4, 3))
    hkl[0, :] = [1,1,1]
    hkl[1, :] = [2,0,0]
    hkl[2, :] = [2,2,0]
    hkl[3, :] = [3,1,1]

    s_dir = np.asarray([0.0,0.0,1.0])

    # Crystal system and lattice parameters, using the same names / parameters as
    # ExaConstit's in-situ light-up options (laue_type and lattice_params) with any
    # angles in radians:
    #   cubic -> a; hexagonal / trigonal / tetragonal -> a, c; rhombohedral -> a, alpha;
    #   orthorhombic -> a, b, c; monoclinic -> a, b, c, beta;
    #   triclinic -> a, b, c, alpha, beta, gamma
    lattice_type = args["lattice_type"]
    lattice_params = args["lattice_params"]
    distance_tolerance = np.deg2rad(args["distance_tolerance_deg"])

    top = fh.read('Element Volumes' , block_id = 0)

    # If we want per element quantities then uncomment below block
    # elem_vols = np.empty((steps, conshape[0]))
    # strains = np.empty((steps, conshape[0], 6))
    # in_fibers = np.zeros((hkl.shape[0], steps, strains.shape[1]), dtype=bool)
    # direct_stiffness = np.zeros((steps - 1, conshape[0]))
    # tay_fact = np.zeros((steps - 1, conshape[0]))
    # eps_rate = np.zeros((steps - 1, conshape[0]))

    lattice_strains = np.zeros((hkl.shape[0], steps))
    lattice_vols = np.zeros_like(lattice_strains)
    lattice_dir_stiff = np.zeros((hkl.shape[0], steps-1))
    lattice_tay_fact = np.zeros((hkl.shape[0], steps-1))
    lattice_eps_rate = np.zeros((hkl.shape[0], steps-1))

    total_volume = np.zeros(steps)

    print("Processing all variables")
    for ii in range(nranks):
        print("Starting rank update: " + str(ii + 1))
        isize = con1d[ii].shape[0] * conshape[1]

        # Read all of the data in
        ev_local = np.ascontiguousarray(fh.read('Element Volumes', start = [0], count = [isize], step_selection = [0 , steps] , block_id = ii).reshape((steps, isize))[:, con1d[ii]])

        # Provide info later related to RVE size so can see how many elements are
        # actually used in the fiber calculations
        total_volume += np.sum(ev_local, axis=1)

        xtal_oris_local = arr = np.ascontiguousarray(fh.read('Crystal Orientations', start = [0, 0], count = [isize, 4], step_selection = [0 , steps] , block_id = ii).reshape((steps, isize, 4))[:, con1d[ii], :])

        elas_strain_local = np.ascontiguousarray(fh.read('Elastic Strains', start = [0, 0], count = [isize, 6], step_selection = [0 , steps] , block_id = ii).reshape((steps, isize, 6))[:, con1d[ii], :])

        stress_local = np.ascontiguousarray(fh.read('Cauchy Stress', start = [0, 0], count = [isize, 6], step_selection = [0 , steps - 1] , block_id = ii).reshape((steps - 1, isize, 6))[:, con1d[ii], :])

        top = fh.read('Shearing Rate' , block_id = 0)
        gdots_local = np.ascontiguousarray(fh.read('Shearing Rate', start = [0, 0], count = [isize, top.shape[1]], step_selection = [0 , steps - 1] , block_id = ii).reshape((steps - 1, isize, top.shape[1]))[:, con1d[ii], :])

        # ExaCMech's equivalent plastic strain rate, used for the Taylor factor the same way the in-situ light-up does
        dpeff_local = np.ascontiguousarray(fh.read('Equivalent Plastic Strain Rate', start = [0], count = [isize], step_selection = [0 , steps - 1] , block_id = ii).reshape((steps - 1, isize))[:, con1d[ii]])

        in_fibers_local = np.zeros((hkl.shape[0], steps, elas_strain_local.shape[1]), dtype=bool)

        xlup.calc_within_fibers(xtal_oris_local, s_dir, hkl, lattice_params, distance_tolerance, in_fibers_local, lattice_type=lattice_type)
        in_fiber_local1 = np.ascontiguousarray(in_fibers_local[:,1:steps,:])
        ev_local1 = np.ascontiguousarray(ev_local[1:steps,:])

        # All of our local calculations. The per_rank_update = True argument makes these
        # add volume weighted sums so every rank's block can be accumulated; they're
        # divided by the in-fiber volumes once all blocks have been processed below.
        # We're already in the sample frame as ExaConstit as of v0.9 automatically converts it for us
        # xlup.strain_lattice2sample(xtal_oris_local, elas_strain_local)
        xlup.calc_lattice_strains(elas_strain_local, s_dir, ev_local, in_fibers_local, lattice_strains, lattice_vols, True)
        xlup.calc_directional_stiffness_lattice_fiber(stress_local, elas_strain_local[1:steps,:,:], lattice_dir_stiff, ev_local1, in_fiber_local1, True)
        xlup.calc_taylor_factors_lattice_fiber(gdots_local, dpeff_local, lattice_tay_fact, lattice_eps_rate, ev_local1, in_fiber_local1, True)

        # If we want per element quantities then uncomment below block
        # direct_stiffness_local = np.zeros_like(ev_local[1:steps, :])
        # tay_fact_local = np.zeros_like(ev_local[1:steps, :])
        # eps_rate_local = np.zeros_like(ev_local[1:steps, :])
        # xlup.calc_directional_stiffness(stress_local, elas_strain_local[1:steps,:,:], direct_stiffness_local)
        # xlup.calc_taylor_factors(gdots_local, dpeff_local, tay_fact_local)
        # eps_rate_local = dpeff_local

        # elem_vols[:, index[ii,0]:index[ii,1]] = ev_local
        # strains[:, index[ii,0]:index[ii,1], :] = elas_strain_local
        # in_fibers[:, :, index[ii,0]:index[ii,1]] = in_fibers_local
        # direct_stiffness[:, index[ii,0]:index[ii,1]] = direct_stiffness_local
        # tay_fact[:, index[ii,0]:index[ii,1]] = tay_fact_local
        # eps_rate[:, index[ii,0]:index[ii,1]] = eps_rate_local

    fh.close()

    # Print data out and then update all of them for the mean values
    print("HKLs used:")
    print(hkl.T)
    print("Total Volume")
    print(total_volume)
    print("Lattice strains:")
    lattice_strains = lattice_strains / lattice_vols
    print(lattice_strains.T)
    print("Lattice taylor factor:")
    lattice_tay_fact = lattice_tay_fact / lattice_vols[:,1:steps]
    print(lattice_tay_fact.T)
    print("Lattice equivalent plastic strain rate:")
    lattice_eps_rate = lattice_eps_rate / lattice_vols[:,1:steps]
    print(lattice_eps_rate.T)
    print("Lattice directional stiffness:")
    lattice_dir_stiff = lattice_dir_stiff / lattice_vols[:,1:steps]
    print(lattice_dir_stiff.T)
    print("Lattice volumes:")
    print(lattice_vols.T)

    out_basename = args["out_basename"]
    out_lattice_quants = out_basename + "lattice_avg_quants"
    out_file = os.path.join(out_dir, out_lattice_quants)
    np.savez_compressed(out_file, lattice_strains=lattice_strains, lattice_tay_fact=lattice_tay_fact, lattice_eps_rate=lattice_eps_rate, lattice_dir_stiff=lattice_dir_stiff, lattice_vols=lattice_vols, hkls=hkl, total_volume=total_volume)

    # If we want per element quantities then uncomment below block to save off
    # in_fibers = in_fibers.astype(np.uint8)
    # out_lightup_processed = out_basename + "light_up_processed.bp"
    # out_file = os.path.join(out_dir, out_lightup_processed)

    # with adios2.Stream(out_file, "w") as s:
    #     s.write("ElementVolumes", elem_vols, shape=elem_vols.shape, start=[0,0], count=elem_vols.shape)
    #     s.write("HKLs", hkl, shape=hkl.shape, start=[0,0], count=hkl.shape)
    #     s.write("FiberElements", in_fibers, shape=in_fibers.shape, start=[0,0,0], count=in_fibers.shape)
    #     s.write("Strains", strains, shape=strains.shape, start=[0,0,0], count=strains.shape)
    #     s.write("DirectionalModulus", direct_stiffness, shape=direct_stiffness.shape, start=[0,0], count=direct_stiffness.shape)
    #     s.write("TaylorFactor", tay_fact, shape=tay_fact.shape, start=[0,0], count=tay_fact.shape)
    #     s.write("EquivalentPlasticStrainRate", eps_rate, shape=eps_rate.shape, start=[0,0], count=eps_rate.shape)

    tf_total = time.time()
    print('%.3f seconds to process %s.' % (tf_total - ts_total, "all items"))

if (__name__ == '__main__') :
    
    parser = argparse.ArgumentParser(
        description = 'Extract specified variables from ExaConstit ADIOS2 outputs.'
        )
    parser.add_argument('-in_dir' ,
                        help = 'path to exaconstit.bp' ,
                        type = str,
                        default='./example_lightup.bp',
                        )
    parser.add_argument('-out_dir' ,
                        help = 'directory to save script outputs' ,
                        type = str,
                        default='./outdir')
    parser.add_argument('-out_basename' ,
                        help = 'basename for outputs typically would be rve name with underscore after it' ,
                        type = str,
                        default='rve_')
    parser.add_argument('-nranks' ,
                        help = 'number of resource sets used for the simulation (IMPORTANT)' ,
                        type = int,
                        default=int(1))
    parser.add_argument('-lattice_type' ,
                        help = 'crystal system (same names as ExaConstit laue_type): cubic, hexagonal, trigonal, rhombohedral, tetragonal, orthorhombic, monoclinic, triclinic' ,
                        type = str,
                        default='cubic')
    parser.add_argument('-lattice_params' ,
                        help = 'lattice parameters for the crystal system with any angles in radians (cubic: a; hexagonal/trigonal/tetragonal: a c; rhombohedral: a alpha; orthorhombic: a b c; monoclinic: a b c beta; triclinic: a b c alpha beta gamma)' ,
                        type = float,
                        nargs = '+',
                        default=[3.60])
    parser.add_argument('-distance_tolerance_deg' ,
                        help = 'angular tolerance in degrees for an orientation to be considered in a fiber' ,
                        type = float,
                        default=5.0)

    args = parser.parse_args()
    print(args)

    args_dict = vars(args)
    fiber_calc_ranks(args_dict)

