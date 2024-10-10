Module wrf-stilt-utils
======================
#========================
# wrf-stilt-utils.py 
#========================

Functions
---------

`locate_trajectory_boundary_points(traj_fname='', save_dir='', bbox=[], write_files=False)`
    Description:        Finds the place in an RDS file called traj_fname from STILT that a particle leaves a 
                        bounding box `bbox`
    Inputs:
        traj_fname      Filename of RDS file from STILT that gives the particle information
        save_dir        Directory where boundary H5 files are saved
        bbox            List [Longitude Lower Bound, Longitude Upper Bound, Latitude Lower Bound, Latitude Upper Bound]
        write_files     Boolean that controls whether files get written out as part of the routine
    
    Returns:
        part            Dict (keys): time (t), particle ID (ind), latitude (lat), longitude (lon), and altitude (zagl)
        bnd             Dict (keys): x, y, z, t of intersection, observation time (obs_t)

`run_function_in_parallel(fun, args_list)`
    Description:       Generic function that runs any function over a set of CPUs -
                       note that this uses concurrent.futures.PoolProcessExecutor due
                       to multithreading issues with rpy2
    Inputs:
        fun            Python function
        args_list      List of dictionaries with keys set to match the input variable
                       argument names for the function fun

`sample_wrfout_met_profile(wrf_domain='d01', wrf_path='', var_dict={})`
    Description:        This function finds the wrfout files closest in time to a sample and
                        then locates the gridbox closest to the sample location from which it 
                        pulls the variables of interest.
    
    Inputs:
        wrf_domain      A string like d01 or d02 - controls which domain you sample from
        wrf_path        Where the wrfout files are stored
        var_dict        Contains the physical locations and times to sample, can also contain 
                        vertical levels if vertical interpolation to preset levels is implied
                        Example: {'lat':[30],'lon':[-75],'datetime':datetime(2023,7,1,18,30),'levs':[10,100]}
    
    Returns:
        var_dict        Augmented dictionary with variables from the wrfout file of interest on levels

`sample_wrfout_single_receptor(wrf_domain='d01', boundary_filename='', overwrite=False)`
    Description:            Finds the wrfout file with the location/time of a particle and 
                            extracts its CO2 and CH4 at the nearest model level with KDTree
                            and writes the information to a file.
    Inputs:
        wrf_domain:         wrfout domain - "d01"
        boundary_filename:  the boundary point file name - contains the location and time
                            where each particle hit the boundary
        overwrite           True means that any existing files are overwitten
    Returns:                None
