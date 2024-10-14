"""
#========================
# wrf-stilt-utils.py 
#========================
"""

from h5py import File
import netCDF4 as nc
import glob,sys,os,pdb,concurrent.futures,time
import numpy as np
import datetime as dt
from scipy.interpolate import interp1d
from pykdtree.kdtree import KDTree
import rpy2.robjects as ro
from rpy2.robjects import conversion,default_converter

def sample_wrfout_met_profile(wrf_domain='d01',wrf_path='',var_dict={}):
    """
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
    """
    files = sorted(glob.glob(f'{wrf_path}/*{wrf_domain}*'))
    wrfdatetime = [dt.datetime.strptime(fi.split(f'{wrf_domain}_')[-1],'%Y-%m-%d_%H:%M:%S') for fi in files]
    try:
        sample_lat = np.array(var_dict['lat'][:])
        sample_lon = np.array(var_dict['lon'][:])
        sample_dt = np.array(var_dict['datetime'][:])
    except KeyError:
        print("var_dict must contain at least one longitude, latitude, and datetime")
        return var_dict

    n_samples = len(sample_lat)
    wrf_files = []
    for i in range(n_samples):
        wrf_files.append(f"{wrf_path}/wrfout_{wrf_domain}_{sample_dt[i].strftime('%Y-%m-%d_%H')}:00:00")
    # Only loop over the unique filenames to save I/O
    unique_wrf_files = sorted(np.array(list(set(wrf_files))))

    f = nc.Dataset(wrf_files[0],'r')
    for v in var_dict:
        if v in ['lat','lon','datetime','levs']: continue
        if 'levs' in var_dict.keys():
            interp_levs = var_dict['levs']
            n_levs = len(interp_levs)
        else:
            n_levs = f['P'][:].shape[1]
        var_dict[v] = np.zeros((n_samples,n_levs))
    for fi in unique_wrf_files:
        f = nc.Dataset(fi,'r')
        wrf_lat = f['XLAT'][:][0]
        wrf_lon = f['XLONG'][:][0]
        tree = KDTree(np.c_[wrf_lon.ravel(),wrf_lat.ravel()])

        time_inds = np.where([fl == fi for fl in wrf_files])[0]
        part_lat = sample_lat[time_inds]
        part_lon = sample_lon[time_inds]
        part_points = np.float32(np.c_[part_lon,part_lat])

        dd,ii = tree.query(part_points,k=1,eps=0.0)
        lat_inds,lon_inds = np.unravel_index(ii,wrf_lat.shape,order='C')
        wrf_gph = (f['PH'][:] + f['PHB'][:])[0][:,lat_inds,lon_inds]
        wrf_z = wrf_gph/9.8
        wrf_zsurf = f['HGT'][:][0,lat_inds,lon_inds]

        for v in var_dict.keys():
            if 'levs' in var_dict.keys(): vert_interp = True
            if v in ['lat','lon','datetime','levs']: continue
            if v.lower() == 'pressure':
                vwrf_smp = f['P'][:][0,:,lat_inds,lon_inds] + f['PB'][:][0,:,lat_inds,lon_inds]
            elif v.lower() == 'psfc':
                vwrf_smp = f['PSFC'][:][0,lat_inds,lon_inds]
                vert_interp = False
            else:
                try:
                    vwrf = f[v][0]
                except KeyError:
                    print(f"{v} not defined")
                    continue
                if len(vwrf.shape) == 2:
                    vwrf_smp = vwrf[lat_inds,lon_inds]
                    vert_interp = False
                    for i,ind in enumerate(time_inds):
                        var_dict[v][ind,0] = vwrf_smp[i]
                    continue
                else:
                    vwrf_smp = vwrf[:,lat_inds,lon_inds]
            if vert_interp:
                for i,ind in enumerate(time_inds):
                    intpf = interp1d(wrf_z[:-1,i]-wrf_zsurf[i],vwrf_smp[i])
                    var_dict[v][ind] = intpf(interp_levs)
            else:
                for i,ind in enumerate(time_inds):
                    var_dict[v][ind] = vwrf_smp[i]

    return var_dict

def sample_wrfout_point_ghg(wrf_domain='d01',wrf_path='',boundary_filename='',overwrite=False):
    """
    Description:            Finds the wrfout file with the location/time of a particle and 
                            extracts its CO2 and CH4 at the nearest model level with KDTree
                            and writes the information to a file. Assumes hourly wrf output.
    Inputs:
        wrf_domain:         wrfout domain - "d01"
        wrf_path:           directory where wrfout files are stored
        boundary_filename:  the boundary point file name - contains the location and time
                            where each particle hit the boundary
        overwrite           True means that any existing files are overwitten
    Returns:
        bc                  dictionary containing the boundary co2 and ch4 from wrf as well
                            as the wrf lat,lon,z closest to the sample
    """
    wrf_bnd_fname = boundary_filename.split('.h5')[0]+'_wrf'+wrf_domain+'.h5'
    if os.path.exists(wrf_bnd_fname):
        if overwrite==True:
            os.remove(wrf_bnd_fname)
        else:
            print('Boundary GHG file already exists, skipping.')
            return

    with File(boundary_filename,'r') as loc_f:
        alt = loc_f['particle_altitude'][:]
        lon = loc_f['particle_longitude'][:]
        lat = loc_f['particle_latitude'][:]
        t = loc_f['particle_time'][:]
        obs_t = (dt.datetime.strptime(loc_f.attrs['obs_time'][:], \
                '%Y-%m-%d %H:%M:%S UTC')-dt.datetime(1970,1,1)).total_seconds()
        loc_f.close()

    #==================================================
    #wrfout_prefix = '/work2/07655/tg869546/stampede3/nyc-chem/2023/wrfout/'
    part_t = np.array([dt.datetime(1970,1,1) + dt.timedelta(seconds=ti) for ti in t])
    wrf_files = np.array([f"{wrf_path}/wrfout_{wrf_domain}_{part_t[ip].strftime('%Y-%m-%d_%H')}:00:00" \
        for ip in range(len(part_t))])
    # Only loop over the unique filenames to save I/O
    unique_wrf_files = sorted(np.array(list(set(wrf_files))))
    #===================================================
    bc = {}
    for v in ['lat','lon','z','co2','ch4']:
        bc[v] = np.zeros(lat.shape)
    bc['t'] = ['' for i in range(len(lat))] # WRFOUT file time
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
        bc['lat'][part_inds] = wrf_lat[lat_inds,lon_inds]
        bc['lon'][part_inds] = wrf_lon[lat_inds,lon_inds]
        for i in part_inds:
            bc['t'][i] = fi.split('/')[-1]

        sub_z = wrf_z[:,lat_inds,lon_inds] - wrf_zsurf[lat_inds,lon_inds]
        z_inds = np.array([np.argmin((part_z[ip]-sub_z[:,ip])**2) for ip in range(len(part_inds))])
        bc['z'][part_inds] = wrf_z[z_inds,lat_inds,lon_inds] - wrf_zsurf[lat_inds,lon_inds]

        co2 = np.zeros(len(part_inds))
        for co2_v in ['ANT','BIO','OCE']:
            co2 += f['CO2_'+co2_v][0][z_inds,lat_inds,lon_inds]-f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
        co2 += f['CO2_BCK'][0][z_inds,lat_inds,lon_inds]
        bc['co2'][part_inds] = co2[:]

        ch4 = np.zeros(len(part_inds))
        for ch4_v in ['ANT','BIO']:
            ch4 += f['CH4_'+ch4_v][0][z_inds,lat_inds,lon_inds]-f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
        ch4 += f['CH4_BCK'][0][z_inds,lat_inds,lon_inds]
        bc['ch4'][part_inds] = ch4[:]
    bc['part_lat'] = lat[:]
    bc['part_lon'] = lon[:]
    bc['part_z'] = alt[:]
    bc['part_t'] = t[:]
    with File(wrf_bnd_fname,'w') as loc_f:
        loc_f.create_dataset('part_lat',data=lat[:])
        loc_f.create_dataset('part_lon',data=lon[:])
        loc_f.create_dataset('part_z',data=alt[:])
        loc_f.create_dataset('part_t',data=t[:])
        loc_f.create_dataset('wrf_lat',data=bc['lat'][:])
        loc_f.create_dataset('wrf_lon',data=bc['lon'][:])
        loc_f.create_dataset('wrf_z',data=bc['z'][:])
        loc_f['wrf_time'] = bc['t'][:]
        loc_f.create_dataset('wrf_co2',data=bc['co2'][:])
        loc_f.create_dataset('wrf_ch4',data=bc['ch4'][:])
        loc_f.close()
    return bc

def locate_trajectory_boundary_points(traj_fname='',save_dir='',bbox=[],write_files=False):
    """
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
    """
    lon_lb,lon_ub,lat_lb,lat_ub = bbox[:]

    with conversion.localconverter(default_converter):
#Trajectories are saved as RDS File
        readRDS = ro.r['readRDS']
        rdf = readRDS(traj_fname)
#----
# Particle information
        part = {}
        part['t'] = np.array(rdf[2][0])*60  # t = time in seconds since release (from minutes in RDF file)
        part['ind'] = np.array(rdf[2][1])   # ind = particle ID
        part['lat'] = np.array(rdf[2][3])   # lat/lon/zagl = physical location of particle at a given time step
        part['lon'] = np.array(rdf[2][2])   # lat/lon/zagl = physical location of particle at a given time step
        part['zagl'] = np.array(rdf[2][4])  # lat/lon/zagl = physical location of particle at a given time step
        n_traj = int(part['ind'].max())     # number of trajectories
#----

#----
# Find boundary intersection
    bnd_inds = []                           # index where particles first hit the boundary of the bbox
    for i in range(1,n_traj+1):
        pt_inds = np.where(part['ind'] == i)[0]
        bind = np.where((part['lon'][pt_inds] >= lon_lb)*(part['lon'][pt_inds] <= lon_ub)*(part['lat'][pt_inds] <= lat_ub)*(part['lat'][pt_inds] >= lat_lb))[0]
        if len(np.where(np.diff(bind) > 1)[0]) == 0:
            bd = bind[-1]
        else:
            bd = bind[np.where(np.diff(bind) > 1)][0]
        bnd_inds.append(bd)

    rec = {}
    for ky in ['lat','lon','zagl','ind','t']:
        rec[ky] = np.nan*np.zeros((n_traj,max(bnd_inds)+1))

    rec['last_ind'] = []
    for i in range(n_traj):
        pt_inds = np.where (part['ind'] == i+1)[0]
        for ky in ['lat','lon','zagl','ind','t']:
            if len(part[ky]) == 0: continue
            rec[ky][i,:bnd_inds[i]] = part[ky][pt_inds][:bnd_inds[i]]
        rec['last_ind'].append(bnd_inds[i])
#----

#----
# Write out boundary points - take the mean of the last 5 points to reduce noise
    inds = rec['last_ind'][:]
    bnd = {}
    bnd['x'] = np.array([rec['lon'][i,inds[i]-5:inds[i]].mean() for i in range(n_traj)])
    bnd['y'] = np.array([rec['lat'][i,inds[i]-5:inds[i]].mean() for i in range(n_traj)])
    bnd['z'] = np.array([rec['zagl'][i,inds[i]-5:inds[i]].mean() for i in range(n_traj)])
    bnd['t'] = np.array([rec['t'][i,inds[i]-5:inds[i]].mean() for i in range(n_traj)]) #time in seconds since release
    bnd['obs_t'] = rdf[1][0][0] #(dt.datetime(1970,1,1) + dt.timedelta(seconds=rdf[1][0][0])).strftime('%Y-%    m-%d %H:%M:%S UTC')
    bnd['part_t'] = bnd['obs_t'] + bnd['t'] # Seconds since 1970,1,1

    if write_files:
        save_fname = save_dir+traj_fname.split('/')[-1].split('.rds')[0]+'_bnd.h5'
        f_out = File(save_fname,'w')
        f_out.attrs['obs_time'] = (dt.datetime(1970,1,1) + dt.timedelta(seconds=rdf[1][0][0])).strftime('%Y-%m-%d %H:%M:%S UTC')
        f_out.create_dataset('particle_time',data=bnd['part_t'][:])
        f_out['particle_time'].attrs['units'] = 'Seconds since 1/1/1970'
        f_out.create_dataset('particle_latitude',data=bnd['y'][:])
        f_out.create_dataset('particle_longitude',data=bnd['x'][:])
        f_out.create_dataset('particle_altitude',data=bnd['z'][:])
        f_out.close()
#----
    return part,bnd

def run_function_in_parallel(fun,args_list):
    """
    Description:       Generic function that runs any function over a set of CPUs -
                       note that this uses concurrent.futures.PoolProcessExecutor due
                       to multithreading issues with rpy2
    Inputs:
        fun            Python function
        args_list      List of dictionaries with keys set to match the input variable
                       argument names for the function fun
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to be executed in parallel with named arguments
        future_to_args = {executor.submit(fun, **args): args for args in args_list}

        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                print(f"Function with args {args} finished successfully!")
            except Exception as exc:
                print(f"Function with args {args} generated an exception: {exc}")


