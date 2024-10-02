#========================
# sample_wrfout.py
# 
# purpose: samples wrfout files to pull out boundary conditions for STILT trajectories
# author: Sean Crowell
# date: Sept 19, 2024
# input: boundary point HDF file from compile_boundary_points.py script
# output: boundary point HDF file augmented with wrfout samples
#
# V0: works with Xiao-Ming Hu's files (moved to old_samplers)
# V1: works with wrfout files in a single directory
#========================

from h5py import File
import netCDF4 as nc
import glob,sys,os,pdb,concurrent.futures,time
import numpy as np
import datetime as dt
import scipy
from pykdtree.kdtree import KDTree

def sample_wrfout_single_receptor(domain='d01',filename=''):
    with File(filename,'r') as loc_f:
        alt = loc_f['particle_altitude'][:]
        lon = loc_f['particle_longitude'][:]
        lat = loc_f['particle_latitude'][:]
        t = loc_f['particle_time'][:]
        obs_t = (dt.datetime.strptime(loc_f.attrs['obs_time'][:],'%Y-%m-%d %H:%M:%S UTC')-dt.datetime(1970,1,1)).total_seconds()
        loc_f.close()
    #==================================================
    wrfout_prefix = '/work2/07655/tg869546/stampede3/nyc-chem/2023/wrfout/'
    part_t = np.array([dt.datetime(1970,1,1) + dt.timedelta(seconds=ti) for ti in t])
    # find all of the files in Xiao-Ming's directory that match up with the trajectories 
    wrf_files = np.array([wrfout_prefix+'/wrfout_'+domain+f'_{part_t[ip].strftime('%Y-%m-%d_%H')}:00:00' for ip in range(len(part_t))])
    # Only loop over the unique filenames to save I/O
    unique_wrf_files = sorted(np.array(list(set(wrf_files))))
    #===================================================

    bc_lat = np.zeros(lat.shape)# NN lat of WRF model
    bc_lon = np.zeros(lat.shape)  # NN lon of WRF model
    bc_z = np.zeros(lat.shape)    # NN altitude of WRF model
    bc_co2 = np.zeros(lat.shape)  # CO2 at WRF 3D NN gridbox
    bc_ch4 = np.zeros(lat.shape)  # CH4 at WRF 3D NN gridbox
    bc_t = ['' for i in range(len(lat))] # WRFOUT file time
    for fi in unique_wrf_files:
        f = nc.Dataset(fi,'r')
        wrf_lat = f['XLAT'][:][0]
        wrf_lon = f['XLONG'][:][0]
        wrf_p = (f['PH'][:] + f['PHB'][:])[0]
        wrf_z = wrf_p/9.8
        wrf_zsurf = f['HGT'][:][0]
        wrf_co2 = f['CO2_BCK'][0]
        for co2_v in ['ANT','BIO','OCE']:
            wrf_co2 += f['CO2_'+co2_v][0]-f['CO2_BCK'][0]
        wrf_ch4 = f['CH4_BCK'][0]
        for ch4_v in ['ANT','BIO']:
            wrf_ch4 += f['CH4_'+ch4_v][0]-f['CH4_BCK'][0]

        tree = KDTree(np.c_[wrf_lon.ravel(),wrf_lat.ravel()])

        part_inds = np.where(wrf_files == fi)[0]
        part_z = alt[part_inds]
        part_lat = lat[part_inds]
        part_lon = lon[part_inds]
        part_points = np.float32(np.c_[part_lon,part_lat])

        dd,ii = tree.query(part_points,k=1,eps=0.0)
        lat_inds,lon_inds = np.unravel_index(ii,wrf_lat.shape,order='C')
        bc_lat[part_inds] = wrf_lat[lat_inds,lon_inds]
        bc_lon[part_inds] = wrf_lon[lat_inds,lon_inds]
        for i in part_inds:
            bc_t[i] = fi.split('/')[-1]

        sub_z = wrf_z[:,lat_inds,lon_inds] - wrf_zsurf[lat_inds,lon_inds]
        z_inds = np.array([np.argmin((part_z[ip]-sub_z[:,ip])**2) for ip in range(len(part_inds))])
        bc_z[part_inds] = wrf_z[z_inds,lat_inds,lon_inds] - wrf_zsurf[lat_inds,lon_inds]

        co2 = np.zeros(len(part_inds))
        for co2_v in ['ANT','BIO','OCE']:
            co2 += f['CO2_'+co2_v][0][z_inds,lat_inds,lon_inds]-f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
        co2 += f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
        bc_co2[part_inds] = co2[:]

        ch4 = np.zeros(len(part_inds))
        for ch4_v in ['ANT','BIO']:
            ch4 += f['CH4_'+ch4_v][0][z_inds,lat_inds,lon_inds]-f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
        ch4 += f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
        bc_ch4[part_inds] = ch4[:]
    wrf_bnd_fname = filename.split('.h5')[0]+'_wrf'+domain+'.h5'
    if os.path.exists(wrf_bnd_fname): os.remove(wrf_bnd_fname)
    with File(wrf_bnd_fname,'w') as loc_f:
        loc_f.create_dataset('part_lat',data=lat[:])
        loc_f.create_dataset('part_lon',data=lon[:])
        loc_f.create_dataset('part_z',data=alt[:])
        loc_f.create_dataset('part_t',data=t[:])
        loc_f.create_dataset('wrf_lat',data=bc_lat[:])
        loc_f.create_dataset('wrf_lon',data=bc_lon[:])
        loc_f.create_dataset('wrf_z',data=bc_z[:])
        loc_f['wrf_time'] = bc_t[:]
        loc_f.create_dataset('wrf_co2',data=bc_co2[:])
        loc_f.create_dataset('wrf_ch4',data=bc_ch4[:])
        loc_f.close()

def run_functions_in_parallel(args_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to be executed in parallel with named arguments
        future_to_args = {executor.submit(sample_wrfout_single_receptor, **args): args for args in args_list}

        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                print(f"Function with args {args} finished with result: {result}")
            except Exception as exc:
                print(f"Function with args {args} generated an exception: {exc}")

if __name__ == "__main__":
    domain = sys.argv[1]
    altitude = sys.argv[2]
    flts = ['20230726_F1','20230726_F2','20230728_F1','20230728_F2','20230805_F1','20230809_F1']
    args_list = []
    for flt in flts:
        flt_files = glob.glob(f'bnd_loc/{flt}/*_{altitude}_*_bnd.h5')
        for fi in flt_files:
            args_list.append({'domain':domain,'filename':fi})
    start_time = time.time()
    run_functions_in_parallel(args_list)
    print(f'Execution Time: {time.time()-start_time} seconds')
