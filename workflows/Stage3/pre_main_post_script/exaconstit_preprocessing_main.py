#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:21:16 2020

@author: carson16

Current version: v0.4
This is meant to be used with the demonstration problem. 
It could be run as a standalone script in which users would want to modify
the bottom code which is behind the if main check to be inline with everything.
Although, I've attempted to make it so that if run from this directory that it
should be able to run and do all of the necessary calculations from the
demonstration problem.

Users could also call the functions in this script in another script if they'd like.
If they wanted to that then the bottom portion of things could be an example
of how to set things up so everything runs as we'd expect.
"""

import numpy as np
import pandas as pd

from job_creation import job_scripts_local as job_scripts

import argparse
import subprocess
import os
import glob
import re
import sys
import pickle

barlat_opt = '../postprocessing/barlat_optimize_v2.py'
if not barlat_opt in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(barlat_opt)))
import barlat_optimize_v2 as bo

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def fixEssVals(repl_val):
    repl_val = re.sub("\n+", "", repl_val)
    repl_val = ' '.join(repl_val.split())
    repl_val = repl_val.replace("0. ", "0.0")
    repl_val = repl_val.replace("0.000000e+00", "0.0")
    return repl_val

def exaconstit_preprocess(args, output_directory):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
       - 'ang_spacing' - The angle spacing between points default to 15 degs if not available, note should be an integer increment
       - 'strain_rate' - The desired strain rate for the problem with default value set at 0.001 1/s
    '''
    #%%
    fdirc = os.path.abspath(args["common_file_directory"])
    #%%
    # Output filenames and directory
    fdiro = os.path.abspath(output_directory)
    fout = args["rve_unique_name"]
    fdiro = os.path.join(fdiro, fout, "")
    fdiro = os.path.join(fdiro, "common_files", "")

    rve_unique_name = args["rve_unique_name"]
    ori_out = fout + "_ori.txt"
    gr_out = fout + "_grains.txt"

    if not os.path.exists(fdiro):
        os.makedirs(fdiro)

    # dt_file = args["dt_file_name"]
    # dt_step = args["number_time_steps"]

    #%%
    # Temperatue ranges that simulations were run at
    tempk = args["temperature"]
    prop_files = args["property_file_names"]
    num_props = args["num_properties"]
    state_files = args["state_file_names"]
    num_states = args["num_states"]
    grain_file_name = args["grain_file_name"]
    ngrains = args["ngrains"]
    shortcut_name = args["shortcut_name"]
    rve_base_name = args["rve_base_name"]
    matl_name = args["matl_name"]
    ori_out = args["ori_file_name"]

    if(len(tempk) != len(prop_files)):
        raise ValueError('Temperature input and property file names not of the same length')

    if(len(tempk) != len(state_files)):
        raise ValueError('Temperature input and state file names not of the same length')

    if ("ang_spacing" not in args.keys()):
        ang_spacing = 15.0
    else:
        ang_spacing = args["ang_spacing"]

    if ("strain_rate" not in args.keys()):
        strain_rate = 0.001
    else:
        strain_rate = args["strain_rate"]

    # At this point this is quite manual. We could probably end up automating this construction later on if 
    # we really wanted to

    # The lists that we'll use to make our dataframe
    loading_name = []
    essential_vals = []
    essential_ids = []
    essential_comps = []
    # These will be repeated in a loop
    rve_name = []
    ori_file_name = []
    temperature = []
    mesh_file = []
    lprop_file = []
    nprop = []
    lstate_file = []
    nstate = []
    ldt_file = []
    ntstep = []
    grain_fname = []
    num_grains = []
    shortcut = []
    material_name = []

    loading_cosines_xy = []
    loading_dir_names_xy = []

    for ang in np.arange(0.0, 91.0, ang_spacing):
        ang90 = 90.0 - ang
        name = "x_" + str(int(ang)) + "_y_" + str(int(ang90))
        loading_cosines_xy.append((ang, ang90, 90.0))
        loading_dir_names_xy.append(name)

    loading_cosines_xz = []
    loading_dir_names_xz = []

    for ang in np.arange(ang_spacing, 91.0, ang_spacing):
        ang90 = 90.0 - ang
        name = "x_" + str(int(ang)) + "_z_" + str(int(ang90))
        loading_cosines_xz.append((ang, 90.0, ang90))
        loading_dir_names_xz.append(name)

    loading_cosines_yz = []
    loading_dir_names_yz = []

    for ang in np.arange(ang_spacing, 90.0, ang_spacing):
        ang90 = 90.0 - ang
        name = "y_" + str(int(ang)) + "_z_" + str(int(ang90))
        loading_cosines_yz.append((90.0, ang, ang90))
        loading_dir_names_yz.append(name)

    loading_dir_names_shear = ["shear_xy", "shear_xz", "shear_yz"]


    vel_x = strain_rate * 1.0
    vel_y = strain_rate * 1.0
    vel_z = strain_rate * 1.0

    essential_ids_xz   = str(np.array2string(np.asarray([1, 2, 3, 4, 5]), separator=', '))
    essential_comps_xz = str(np.array2string(np.asarray([3, 1, 2, 3, 1]), separator=', '))
    essential_comps_xz_fx = str(np.array2string(np.asarray([3, 1, 2, 3, 0]), separator=', '))
    essential_vals_xz  = np.asarray([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, vel_z], 
                                    [vel_x, 0.0, 0.0]])

    # rve_unique_name, ori_file_name, ngrains, tempk,
    # prop_file_loc, nprops, state_file_loc, nstates,
    # ess_id_array, ess_comp_array, ess_vals_array,
    # mesh_file_loc

    for iname in range(len(loading_dir_names_xz)):
        arr = np.copy(essential_vals_xz)
        xd, yd, zd = loading_cosines_xz[iname]
        arr[:, 0] = arr[:, 0] * np.cos(np.deg2rad(xd))
        arr[:, 2] = arr[:, 2] * np.cos(np.deg2rad(zd))
        arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
        loading_name.append(loading_dir_names_xz[iname])
        essential_vals.append(arr1d_str)
        if(iname < len(loading_dir_names_xz) - 1):
            essential_comps.append(essential_comps_xz)
        else:
            essential_comps.append(essential_comps_xz_fx)
        essential_ids.append(essential_ids_xz)


    essential_ids_yz   = str(np.array2string(np.asarray([1, 2, 3, 4, 6]), separator=', '))
    essential_comps_yz = str(np.array2string(np.asarray([3, 1, 2, 3, 2]), separator=', '))
    essential_vals_yz  = np.asarray([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, vel_z], 
                                    [0.0, vel_y, 0.0]])

    for iname in range(len(loading_dir_names_yz)):
        arr = np.copy(essential_vals_yz)
        xd, yd, zd = loading_cosines_yz[iname]
        arr[:, 1] = arr[:, 1] * np.cos(np.deg2rad(yd))
        arr[:, 2] = arr[:, 2] * np.cos(np.deg2rad(zd))
        arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
        loading_name.append(loading_dir_names_yz[iname])
        essential_vals.append(arr1d_str)
        essential_ids.append(essential_ids_yz)
        essential_comps.append(essential_comps_yz)

    essential_ids_xy   = str(np.array2string(np.asarray([1, 2, 3, 5, 6]), separator=', '))
    essential_comps_xy = str(np.array2string(np.asarray([3, 1, 2, 1, 2]), separator=', '))
    essential_comps_xy_fy = str(np.array2string(np.asarray([3, 1, 2, 1, 0]), separator=', '))
    essential_comps_xy_fx = str(np.array2string(np.asarray([3, 1, 2, 0, 2]), separator=', '))
    essential_vals_xy  = np.asarray([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [vel_x, 0.0, 0.0], 
                                    [0.0, vel_y, 0.0]])

    for iname in range(len(loading_dir_names_xy)):
        arr = np.copy(essential_vals_xy)
        xd, yd, zd = loading_cosines_xy[iname]
        arr[:, 0] = arr[:, 0] * np.cos(np.deg2rad(xd))
        arr[:, 1] = arr[:, 1] * np.cos(np.deg2rad(yd))
        arr1d_str = str(np.array2string(arr.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))
        loading_name.append(loading_dir_names_xy[iname])
        essential_vals.append(arr1d_str)
        essential_ids.append(essential_ids_xy)
        if(iname == 0):
            essential_comps.append(essential_comps_xy_fy)
        elif(iname < len(loading_dir_names_xy) - 1):
            essential_comps.append(essential_comps_xy)
        else:
            essential_comps.append(essential_comps_xy_fx)

    essential_ids_shear_xy   = str(np.array2string(np.asarray([2, 5]), separator=', '))
    essential_comps_shear_xy = str(np.array2string(np.asarray([7, 7]), separator=', '))
    essential_vals_shear_xy  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, vel_y, 0.0]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

    loading_name.append(loading_dir_names_shear[0])
    essential_vals.append(essential_vals_shear_xy)
    essential_ids.append(essential_ids_shear_xy)
    essential_comps.append(essential_comps_shear_xy)

    essential_ids_shear_xz   = str(np.array2string(np.asarray([2, 5]), separator=', '))
    essential_comps_shear_xz = str(np.array2string(np.asarray([7, 7]), separator=', '))
    essential_vals_shear_xz  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, vel_z]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

    loading_name.append(loading_dir_names_shear[1])
    essential_vals.append(essential_vals_shear_xz)
    essential_ids.append(essential_ids_shear_xz)
    essential_comps.append(essential_comps_shear_xz)

    essential_ids_shear_yz   = str(np.array2string(np.asarray([3, 6]), separator=', '))
    essential_comps_shear_yz = str(np.array2string(np.asarray([7, 7]), separator=', '))
    essential_vals_shear_yz  = str(np.array2string(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, vel_z]), separator=', ', formatter={'float_kind':lambda x: "%.8f" % x}, max_line_width=360))

    loading_name.append(loading_dir_names_shear[2])
    essential_vals.append(essential_vals_shear_yz)
    essential_ids.append(essential_ids_shear_yz)
    essential_comps.append(essential_comps_shear_yz)

    nruns = len(loading_name)

    lloading_name = loading_name.copy()
    lessential_vals = essential_vals.copy()
    lessential_ids = essential_ids.copy()
    lessential_comps = essential_comps.copy()

    for itemp in range(len(tempk)):
        for i in range(nruns):
            rve_name.append(rve_unique_name)
            fho = os.path.join(fdirc, os.path.basename(ori_out))
            ori_file_name.append(fho)
            temperature.append(tempk[itemp])
            fhp = os.path.join(fdirc, os.path.basename(prop_files[itemp]))
            lprop_file.append(fhp)
            fhs = os.path.join(fdirc, os.path.basename(state_files[itemp]))
            lstate_file.append(fhs)
            nprop.append(num_props[itemp])
            nstate.append(num_states[itemp])
            grain_fname.append(grain_file_name)
            num_grains.append(ngrains)
            shortcut.append(shortcut_name)
            material_name.append(matl_name)
        if(itemp > 0):
            lloading_name.extend(loading_name)
            lessential_vals.extend(essential_vals)
            lessential_ids.extend(essential_ids)
            lessential_comps.extend(essential_comps)

    data = {"rve_unique_name" : rve_name, "prop_file_loc" : lprop_file, "nprops" : nprop, "temperature" : temperature,
            "grain_file_name" : grain_fname, "ngrains" : num_grains, "shortcut_name" : shortcut, "matl_name" : material_name,
            "state_file_loc" : lstate_file, "nstates" : nstate, "ori_file_name" : ori_file_name,
            "ess_id_array" : lessential_ids, "ess_comp_array" : lessential_comps, "ess_vals_array" : lessential_vals,
            "loading_name" : lloading_name}

    df = pd.DataFrame(data)

    return df

def exaconstit_job_cli(args, output_file_dir, df):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
    '''
    #%%
    # Simulation directory

    fdirs = os.path.abspath(output_file_dir)
    frve = args["rve_unique_name"]
    fdir_rve = os.path.join(fdirs, frve, "")
    fdirs = os.path.join(fdir_rve, "common_files", "")
    #%%
    # Input job script filename and directory
    fin   = args["input_job_filename"]
    ftoml = args["input_master_toml"]
    fotoml = args["input_output_toml"]
    fdirc = os.path.abspath(args["common_file_directory"])
    fdiri = args["input_job_filedir"]

    if (fdiri == "./"):
        fdiri = os.getcwd()
        os.path.join(fdiri, '')

    #%%
    # Output directory
    fdiro = fdir_rve

    #%%
    # Create all of the necessary symlink files and job script
    fh = os.path.join(fdirc, os.path.basename(fin))
    #%%
    # If we want to be closer to how Themis (LLNL software that operates on top of flux and can automate a lot of this job submission/creation)
    # does things then we'll need to have a csv file with our various test matrix parameters in it.
    # We'll then have the headers act as a key to python dictionary of a list that contains the parameter value 
    # for each test. Next, we'd want a master .toml/option file for our simulations where we have %%key_name%% 
    # through out the file for the various params that need to be modified/filled in from the provided csv file.
    # These parameters will be the output name for the body average or volume values text files. The name
    # should compose of the unique microstructure name, temperature, and loading info. The unique microstructure name
    # will have provided by the ExaCA people. The other parts we'll come up which won't be hard.
    # Other params that need to be changed are the orientation, mesh, property, state, and time step file names. These will largely
    # be constant for a given microstructure. The only one which will vary will be the property file, since the properties vary with temperature.
    # Finally, we'll need to modify the  loading conditions. Currently, the plan is for a total of 21 tests per temperature.
    # The file replacement shouldn't be too bad. Our toml file is fairly small, so we can quickly iterate through it line by line
    # and just do the search for the keys and replace them with the appropriate value for each test case.
    # 
    # Long story short, if we're doing the real workflow and not just a test of the system we'll read in the csv
    # file first before this loop. We'll then know how many tests to run.
    # Then, we'll add a function call in here to generate the correct toml file for each
    # simulation in here. In the end, we can keep the numeric numbering of the directories, since it's not super
    # important or we can rename it as microstructure_name_temperature_incr_#. 
    # Some logic will need to be added to the optimization script to walk through our work flow directory once all
    # simulations are finished and group the necessary simulations together (same microstructure and same temperature). 

    # Create all of the necessary symlink files and job script
    fh = os.path.join(fdirc, os.path.basename(ftoml))

    mtoml = []

    # Read all the data in as a single string
    # should make doing the regex easier
    with open(fh, "rt") as f:
        mtoml = f.read()

    # The csv file should be generated in the pre-processing step
    # as we'll have much of the information already available there
    # The csv should have the following lines:
    # RVE unique name, orientation file name, # of grains
    # temperature (K), property file name, # of properties
    # state file name, # of state variables, essential id array,
    # essential components array, essential_vals array,
    # loading direction name, mesh file location
    #
    # Header for the should be:
    # rve_unique_name, ori_file_name, ngrains, tempk,
    # prop_file_loc, nprops, state_file_loc, nstates,
    # ess_id_array, ess_comp_array, ess_vals_array,
    # loading_name, mesh_file_loc

    nruns = df.shape[0]
    headers = list(df.columns)
    headers.pop(0)

    avg_headers = ["rve_name"]
    if not os.path.exists(fdiro):
        os.makedirs(fdiro)
    for iDir in range(nruns):
        rve_name = df["rve_unique_name"][iDir]
        load_dir_name = df["loading_name"][iDir]
        temp_k = str(int(df["temperature"][iDir]))
        fdiron = fdir_rve
        fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")
        if not os.path.exists(fdironl):
            os.makedirs(fdironl)
        # Create symlink from RVE common files
        for src in glob.glob(os.path.join(fdirs, "*")):
            fh = os.path.join(fdironl, os.path.basename(src))
            if not os.path.exists(fh):
                os.symlink(src, fh)
        # Create symlink from global common files
        for src in glob.glob(os.path.join(fdirc, "*")):
            fh = os.path.join(fdironl, os.path.basename(src))
            if not os.path.exists(fh):
                os.symlink(src, fh)
        toml = mtoml
        for iheader in headers:
            search = "%%" + iheader + "%%"
            repl_val = str(df[iheader][iDir])
            # This line is needed as toml parsers might get mad with just the
            # 0. and not 0.0
            repl_val = fixEssVals(repl_val)
            toml = re.sub(search, repl_val, toml)
        # Now do the avg_stress, avg_pl_work, avg_dp_tensor replacements
        # We always want these to be unique names so if they're moved somewhere
        # else we can always associate them with the correct run
        # therefore, the name contains the rve name, temperature, and loading dir name
        # frve_name+"_"+str(temp)+"_"+loading_dir_names[0]
        ext_name = rve_name +"_" + temp_k + "_" + load_dir_name
        for iheader in avg_headers:
            search = "%%" + iheader + "%%"
            replace = ext_name
            toml = re.sub(search, replace, toml)

        # Output toml file
        fh = os.path.join(fdironl, os.path.basename(fotoml))
        # Check to see if it is a symlink and if so remove the link
        if os.path.islink(fh):
            os.unlink(fh)
        # We can now safely write out the file
        with open(fh, "w") as f:
            f.write(toml)

    return None

def exaconstit_job_generation(input_cases, output_file_dir, pre_process=True, job_creation=False, post_process=False):
    '''
    exaconstit_job_generation takes in a panada dataframe (input_cases) and the desired output file directory for all runs:
       input_cases has the following set of headers that need to be filled out for each RVE
       - 'exaca_input_file_dir' - Directory of ExaCA simulation data
       - 'exaca_input_file' - File name of ExaCA data
       - 'unique_ori_filename' - File name / path of unique orientation file as quaternions used by ExaCA (this is usually a universal file but as a rmat so some minor conversions will need to occur potentially)
       - 'coarsening' - Level of coarsening to apply 
                        (level 1 = original CA voxel mesh, 
                         level 2 = takes 2x2x2 voxels and average that to 1 voxel / grain ID
                         level n = takes nxnxn voxels and average that to 1 voxel / grain ID)
       - 'mesh_generator' - Run the mesh generator
       - 'mesh_generator_dir' - Mesh generator directory
       - 'rve_unique_name' - Unique name for a given microstructure
       - 'temperature' - List of temperatures in Kelvin for simulations to run at
       - 'property_file_names' - List of property file names to be used in simulations, and it should be same length as the temperature list
       - 'num_properties' - Number of properties in the property files
       - 'state_file_names' - List of state file names to be used in simulations, and it should be same length as the temperature list
       - 'num_states' - Number of states variables in the state.txt file
       - 'dt_file_name' - Custom dt time step file we want to use
       - 'number_time_steps' - Number of time steps to be taken
       - 'common_file_directory' - Location of common/shared property/state/dt files for various RVEs 
       - 'input_job_filename' - Input job script filename for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_job_filedir' - Path directory of the input job script
       - 'bsub_jobs' - Create a batch script to submit all of our bsub jobs rather than using flux to manage the job pool
       - 'input_master_toml' - Master option toml file for ExaConstit for which we\'ll use to run all of our jobs
       - 'input_output_toml' - Option toml file name used for ExaConstit for which we\'ll use to run a job
       - 'rtmodel' - Value to use as Solvers.rtmodel in configured options file
    '''
    nrves = len(input_cases.index)
    headers = list(input_cases.columns)
    headers.pop(0)

    scripts_run = []
    rve_test_matrices = []
    rve_test_file = os.path.join(output_file_dir, "rve_test_matrices.pickle")
    post_process_file = os.path.join(output_file_dir, "post_process.pickle")

    if pre_process:
        for irve in range(nrves):
            local_input_cases = input_cases.loc[irve]
            print("Starting preprocessing step")
            rve_test_matrix = exaconstit_preprocess(local_input_cases, output_file_dir)
            print("Starting simulation generation step")
            scripts = exaconstit_job_cli(local_input_cases, output_file_dir, rve_test_matrix)
            scripts_run.append(scripts)
            rve_test_matrices.append(rve_test_matrix.copy())

        with open(rve_test_file, "wb") as f_handle:
            pickle.dump(rve_test_matrices, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(rve_test_file, "rb") as f_handle:
            rve_test_matrices = pickle.load(f_handle)

    # This is not at all efficient way to submit all of the jobs but...
    # it should get the job done for the near term
    # We assume that this script will run submit and or run all of our simulations
    if job_creation:
        job_scripts(input_cases, output_file_dir, rve_test_matrices)
    if post_process:
        post_results = bo.postprocessing(input_cases, output_file_dir, rve_test_matrices, nrves, 
                                        quaternion_file=input_cases["quaternion_file"][0], quat_weights_file=input_cases["quaternion_weights"][0], use_sachs=input_cases["ori_optimization"][0])
        with open(post_process_file, "wb") as f_handle:
            pickle.dump(post_results, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
        

#        for irve in range(nrves):
#            local_input_cases = input_cases.loc[irve]
#            fdirs = os.path.abspath(output_file_dir)
#            frve_name = local_input_cases["rve_unique_name"]
#            fdir_rve = os.path.join(fdirs, frve, "")
#
#            ftime = local_input_cases["dt_file_name"]
#            tempk = local_input_cases["temperature"]
#            bo.postprocessing(frve_name, fdir_rve, ftime, tempk, False)

if __name__ == "__main__":
    # Input parameters for preprocessing/simulation portion of things
    inputs = {
        "rve_unique_name" : [],
        "temperature" : [],
        "grain_file_name" : [],
        "ngrains" : [],
        "shortcut_name" : [],
        "rve_base_name" : [],
        "matl_name" : [],
        "property_file_names" : [],
        "num_properties" : [],
        "state_file_names" : [],
        "num_states" : [],
        "ori_file_name" : [],
        "quaternion_file" : None,
        "quaternion_weights" : None,
        "ori_optimization" : None,
        "common_file_directory" : [],
        "input_job_filename" : [],
        "input_job_filedir" : [],
        "input_master_toml" : [],
        "input_output_toml" : [],
        "strain_rate": [],
        "ang_spacing": [],
        "exaconstit_binary" : [],
        "exaconstit_module_source_file" : [],
        "job_num_nodes" : [],
        "job_walltime"  : [],
        "job_node_cpus" : [],
        "job_node_gpus" : [],
        "job_max_nodes_fail"    : [],
        "job_max_walltime_fail" : [],
        "rve_job_num_nodes" : [],
        "rve_job_num_ranks" : [],
        "rve_job_time" : []
    }

    # Relevant data files we need to run
    path_dir = "./"
    path_dir = os.path.abspath(path_dir)
    output_file_dir = os.path.join(path_dir, "cases", "")

    temperatures = [298.0]
    ang_spacing = 1.0
    strain_rate = 0.001
    property_files = ["props_cp_in625.txt"]
    num_props = [17]
    state_files = ["state_cp.txt"]
    num_states = [24]
    grain_file_name = "grains.txt"
    ngrains = 1000
    shortcut_name = "evptn_FCC_A"
    rve_base_name = "px_in625_deg_1"
    matl_name = "in625"
    # Where ever we have the common files located
    common_file_directory = "../common_simulation_files/"
    common_file_directory = os.path.abspath(common_file_directory)
    input_job_filename = "challenge_problem.bash"
    input_job_filedir = common_file_directory
    bsub_jobs = False
    input_master_toml = "options_master.toml"
    input_output_toml = "options.toml"
    ori_file_name = "quats_1k.txt"
    rtmodel = "CPU"

    inputs["quaternion_file"] = os.path.join(common_file_directory, ori_file_name)
    inputs["ori_optimization"] = True

    num_nodes = 1
    num_resources_per_node = 56
    rve_job_num_ranks = num_nodes * num_resources_per_node

    job_num_nodes = int(1)
    job_node_cpus = int(112)
    job_node_gpus = int(0)
    # Walltime is in minutes
    job_walltime  = int(120.0)
    job_max_nodes_fail = int(10)
    job_max_walltime_fail = int(240.0)
    num_resources_per_node = int(112)
    rve_job_num_nodes = int(1)
    rve_job_num_ranks = rve_job_num_nodes * num_resources_per_node
    rve_job_time = int(10.0)

    # Location of the ExaConstit binary
    exaconstit_binary = os.environ.get("exaconstit_binary", "mechanics")
    # Location of shell script file that will be used to source/load our necessary modules needed to run ExaConstit 
    exaconstit_module_source_file = os.path.abspath("./")

    nruns = 1

    for iruns in range(nruns):
        inputs["rve_unique_name"].append(rve_base_name)
        inputs["grain_file_name"].append(grain_file_name)
        inputs["ngrains"].append(ngrains)
        inputs["shortcut_name"].append(shortcut_name)
        inputs["rve_base_name"].append(rve_base_name)
        inputs["matl_name"].append(matl_name)
        inputs["temperature"].append(temperatures)
        inputs["property_file_names"].append(property_files)
        inputs["num_properties"].append(num_props)
        inputs["state_file_names"].append(state_files)
        inputs["num_states"].append(num_states)
        inputs["ori_file_name"].append(ori_file_name)
        inputs["common_file_directory"].append(common_file_directory)
        inputs["input_job_filename"].append(input_job_filename)
        inputs["input_job_filedir"].append(input_job_filedir)
        inputs["input_master_toml"].append(input_master_toml)
        inputs["input_output_toml"].append(input_output_toml)
        inputs["strain_rate"].append(strain_rate)
        inputs["ang_spacing"].append(ang_spacing)
        #Requirements related to our job
        inputs["exaconstit_binary"].append(exaconstit_binary)
        inputs["exaconstit_module_source_file"].append(exaconstit_module_source_file)
        inputs["job_num_nodes"].append(job_num_nodes)
        inputs["job_walltime"].append(job_walltime)
        inputs["job_node_cpus"].append(job_node_cpus)
        inputs["job_node_gpus"].append(job_node_gpus)
        inputs["job_max_nodes_fail"].append(job_max_nodes_fail)
        inputs["job_max_walltime_fail"].append(job_max_walltime_fail)
        inputs["rve_job_num_nodes"].append(rve_job_num_nodes)
        inputs["rve_job_num_ranks"].append(rve_job_num_ranks)
        inputs["rve_job_time"].append(rve_job_time)

    input_cases = pd.DataFrame(data=inputs)

    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    # Gives the ability to restart things if need be or do post-processing at
    # a later time
    fhdf = os.path.join(output_file_dir, 'uq_stage3_test_matrix.csv')
    input_cases.to_csv(fhdf)

    # Once run all of our jobs should be submitted to the LSF system
    # If we're on a Flux system then we could actually modify things so that
    # we use the python interface to submit and wait on all the jobs
    exaconstit_job_generation(input_cases, output_file_dir, True, False, False)
