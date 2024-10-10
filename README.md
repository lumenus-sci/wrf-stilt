Module wrf-stilt-utils
======================

Functions
---------

`locate_trajectory_boundary_points(traj_fname='', save_dir='', bbox=[], write_files=False)`
:   

`run_function_in_parallel(fun, args_list)`
:   

`sample_wrfout_met_profile(wrf_domain='d01', wrf_path='', var_dict={})`
:   This function finds the wrfout files closest in time to a sample and
    then locates the gridbox closest to the sample location from which it 
    pulls the variables of interest.

`sample_wrfout_single_receptor(wrf_domain='d01', boundary_filename='', overwrite=False)`
:
