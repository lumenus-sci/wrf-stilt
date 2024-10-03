#========================
# find_boundary_points.py
# 
# purpose: Example driver script to sample wrfout files with boundary points from STILT particle trajectories
# author: Sean Crowell
# input: bounding box, trajectory directory, altitude
# output: boundary point HDF file
#
# V0: works with UATAQ RDS files 
#========================

import matplotlib.pyplot as plt
import numpy as np
import io,os,sys,time
from h5py import File
import pandas as pd
import scipy
import glob
import pyreadr
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import datetime as dt
wrf_stilt_utils = __import__("wrf-stilt-utils")

def create_halo_file_list(traj_dir='',altitude=''):
    all_files = glob.glob(f'{traj_dir}/*.rds')
    alts = np.array([fi.split('_')[-2] for fi in all_files])
    alt_inds = np.where(alts == altitude)[0]
    return [all_files[i] for i in alt_inds]

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f'Usage: python find_boundary_points.py lon_lb,lon_ub,lat_lb,lat_ub traj_dir altitude savedir')
        sys.exit()
    bound_box = [np.float32(x) for x in sys.argv[1].split(',')]
    traj_dir = sys.argv[2]
    altitude = str(sys.argv[3])
    savedir = sys.argv[4]

    file_names = create_halo_file_list(traj_dir=traj_dir,altitude=altitude)
    args_list = []
    for fi in file_names:
        args_list.append({'traj_fname':fi,'save_dir':savedir,'bbox':bound_box,'write_files':False})

    start_time = time.time()
    wrf_stilt_utils.run_function_in_parallel(wrf_stilt_utils.locate_trajectory_boundary_points,args_list)
    print(f'Run finished in {time.time()-start_time} seconds')