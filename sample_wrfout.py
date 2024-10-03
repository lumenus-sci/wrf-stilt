#========================
# sample_wrfout.py
#
# purpose: samples wrfout files to pull out boundary conditions for STILT trajectories
# author: Sean Crowell
# input: boundary point HDF file from find_boundary_points.py script
# output: boundary point HDF file augmented with wrfout samples
#
# V0: works with Xiao-Ming Hu's files (moved to old_samplers) - 9/19/2024
# V1: works with wrfout files in a single directory
# V2: added parallel functionality with concurrent.futures - 10/2/2024
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

def create_halo_file_list(domain='d01',altitude=50):
    flts = ['20230726_F1','20230726_F2','20230728_F1','20230728_F2','20230805_F1','20230809_F1']
    args_list = []
    for flt in flts:
        flt_files = glob.glob(f'/scratch/07351/tg866507/halo/bnd/bnd_loc/{flt}/*_{altitude}_*_bnd.h5')
        for fi in flt_files:
            args_list.append({'domain':domain,'filename':fi})
    return args_list

if __name__ == "__main__":
    domain = sys.argv[1]     # WRF domain: d01 or some higher resolution domain
    altitude = sys.argv[2]   # Altitude of receptor used to subset filenames
    args_list = create_halo_file_list(domain=domain,altitude=altitude)
    start_time = time.time()
    run_sampler_in_parallel(args_list)
    print(f'Execution Time: {time.time()-start_time} seconds')