"""
#========================
# wrf-stilt-utils.py 
#========================
"""
import warnings
from h5py import File
import netCDF4 as nc
import glob,sys,os,pdb,concurrent.futures,time
import numpy as np
import datetime as dt
from scipy.interpolate import interp1d
from pykdtree.kdtree import KDTree
import rpy2.robjects as ro
from rpy2.robjects import conversion,default_converter
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, MultiPolygon, Feature

def locate_trajectory_boundary_points(ID='',trajectory_rds_filename='',save_dir='',bbox=None,write_files=False):
    """
    Description:        Finds the place in an RDS file called traj_fname from STILT that a particle leaves a 
                        bounding box `bbox`
    Inputs:
        ID:             unique string for this receptor
        traj_fname      Filename of RDS file from STILT that gives the particle information
        save_dir        Directory where boundary H5 files are saved
        bbox            List [Longitude Lower Bound, Longitude Upper Bound, Latitude Lower Bound, Latitude Upper Bound]
        write_files     Boolean that controls whether files get written out as part of the routine
    
    Returns:                    return_dict consisting of
        receptor_loc_vars       lat, lon, datetime, levs (ZAGL of receptor) of receptor - these are constructed to match 
                                with the more general sample_wrfout_profile function below
        bnd_loc_vars            bnd_lat, bnd_lon, bnd_zagl, bnd_t of intersection
    """
    
    with conversion.localconverter(default_converter):
#Trajectories are saved as RDS File
        readRDS = ro.r['readRDS']
        rdf = readRDS(trajectory_rds_filename)
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
# Initialize return_dict

    return_dict = {}
    return_dict['receptor_loc_vars'] = {'lat':[float(rdf[1][1][0])],'lon':[float(rdf[1][2][0])],
            'datetime':[dt.datetime(1970,1,1) + dt.timedelta(seconds=rdf[1][0][0])],
            'levs':[float(rdf[1][3][0])]}
#
#----

#----
# Find boundary intersection
    bnd_inds = []                           # index where particles first hit the boundary of the bbox
    rec = {}
    bnd = {}
    for i in range(1,n_traj+1):
        pt_inds = np.where(part['ind'] == i)[0]
        points = [Feature(geometry=Point([part['lat'][ind],part['lon'][ind]])) for ind in pt_inds]
        polygons = [bbox for j in range(len(pt_inds))]
        bind = np.where(list(map(boolean_point_in_polygon,points,polygons)))[0]
        if len(bind) < 2 :
            bd = 0 
        elif len(np.where(np.diff(bind) > 1)[0]) == 0:
            bd = bind[-1]
        else:
            bd = bind[np.where(np.diff(bind) > 1)][0]
        bnd_inds.append(bd)

    for ky in ['lat','lon','zagl','ind','t']:
        rec[ky] = np.nan*np.zeros((n_traj,max(bnd_inds)+1))

    rec['last_ind'] = []
    for i in range(n_traj):
        pt_inds = np.where(part['ind'] == i+1)[0]
        for ky in ['lat','lon','zagl','ind','t']:
            if len(part[ky]) == 0: continue
            rec[ky][i,:bnd_inds[i]+1] = part[ky][pt_inds][:bnd_inds[i]+1]
        rec['last_ind'].append(bnd_inds[i])
#----

#----
# Locate boundary points - take the mean of the last 5 points to reduce noise
    inds = rec['last_ind'][:]
    bnd['bnd_lon'] = np.array([rec['lon'][i,inds[i]] for i in range(n_traj)])
    bnd['bnd_lat'] = np.array([rec['lat'][i,inds[i]] for i in range(n_traj)])
    bnd['bnd_zagl'] = np.array([rec['zagl'][i,inds[i]] for i in range(n_traj)])
    t_s = np.array([rec['t'][i,inds[i]] for i in range(n_traj)])
    bnd['bnd_t'] = rdf[1][0][0] + t_s # Seconds since 1970,1,1
    return_dict['bnd_loc_vars'] = bnd
#----

#----
# Write out trajectories to HDF5 files if write_files==True
    if write_files:
        os.makedirs(save_dir,exist_ok=True)
        save_fname = save_dir+trajectory_rds_filename.split('/')[-1].split('.rds')[0]+'.h5'
        if os.path.exists(save_fname):
            os.remove(save_fname)
        f_out = File(save_fname,'w')
        f_out.attrs['obs_time'] = (dt.datetime(1970,1,1) + dt.timedelta(seconds=rdf[1][0][0])).strftime('%Y-%m-%d %H:%M:%S UTC')
        f_out.attrs['obs_lat'] = float(rdf[1][1][0])
        f_out.attrs['obs_lon'] = float(rdf[1][2][0])
        f_out.attrs['obs_alt'] = float(rdf[1][3][0])
        f_out.create_dataset('particle_time',data=rec['t'][:])
        f_out['particle_time'].attrs['units'] = 'Seconds since release'
        f_out.create_dataset('particle_latitude',data=rec['lat'][:])
        f_out.create_dataset('particle_longitude',data=rec['lon'][:])
        f_out.create_dataset('particle_altitude',data=rec['zagl'][:])
        f_out.create_dataset('boundary_index',data=rec['last_ind'][:])
        f_out.close()
#----
    return return_dict

def sample_wrfout_profile(wrf_domain='d01',wrf_path='',var_dict={}):
    """
    Description:        This function finds the wrfout files closest in time to a sample and
                        then locates the gridbox closest to the sample location from which it 
                        pulls the variables of interest.

    Inputs:
        wrf_domain      A string like d01 or d02 - controls which wrfout domain files you sample from
        wrf_path        Where the wrfout files are stored
        var_dict        Contains the physical locations and times to sample, can also contain 
                        vertical levels if vertical interpolation to preset levels in meters AGL is implied
                        Example: {'lat':[30],'lon':[-75],'datetime':datetime(2023,7,1,18,30),'levs':[10,100]}

    Returns:
        var_dict        Augmented dictionary with variables from the wrfout file of interest on vertical levels
    """
    files = sorted(glob.glob(f'{wrf_path}/*{wrf_domain}*'))
    wrfdatetime = [dt.datetime.strptime(fi.split(f'{wrf_domain}_')[-1],'%Y-%m-%d_%H:%M:%S') for fi in files]
    try:
        sample_lat = np.array(var_dict['lat'])
        sample_lon = np.array(var_dict['lon'])
        sample_dt = np.array(var_dict['datetime'])
    except KeyError:
        print("var_dict must contain at least one longitude, latitude, and datetime")
        return var_dict

#----
#   Find all of the wrfout files needed for the set of points and datetimes
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

        time_inds = np.where([fl == fi for fl in wrf_files])[0]
        part_lat = sample_lat[time_inds]
        part_lon = sample_lon[time_inds]
        part_points = np.float32(np.c_[part_lon,part_lat])
        
        # Define the KDTree to find the nearest neighbor latitude/longitude
        tree = KDTree(np.c_[wrf_lon.ravel(),wrf_lat.ravel()])
        dd,ii = tree.query(part_points,k=1,eps=0.0)
        lat_inds,lon_inds = np.unravel_index(ii,wrf_lat.shape,order='C')
        wrf_gph = (f['PH'][:] + f['PHB'][:])[0][:,lat_inds,lon_inds]
        wrf_z = wrf_gph/9.8
        wrf_zsurf = f['HGT'][:][0,lat_inds,lon_inds]

        # Sample the model at the lat/lon and vertical levels for each variable
        # If a variable name string is not in the wrfout file, a new elif statement
        # for computing it from variables in the file must be added below.
        for v in var_dict.keys():
            if 'levs' in var_dict.keys(): vert_interp = True
            if v in ['lat','lon','datetime','levs']: continue
            if v.lower() == 'pressure': 
                vwrf_smp = (f['P'][:][0]+f['PB'][:][0])[:,lat_inds,lon_inds]
            elif v.lower() == 'psfc':
                vwrf_smp = f['PSFC'][:][0][lat_inds,lon_inds]
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
                    vert_inds = np.where((interp_levs >= wrf_z[0,i])*(interp_levs <= wrf_z[-1,i]))[0]
                    var_dict[v][ind] = np.nan*np.zeros(len(interp_levs))
                    intpf = interp1d((wrf_z[:-1,i]-wrf_zsurf[i]).data,vwrf_smp[:,i].data)
                    var_dict[v][ind][vert_inds] = intpf(interp_levs[vert_inds])
            else:
                for i,ind in enumerate(time_inds):
                    var_dict[v][ind] = vwrf_smp[i]

    return var_dict

def stilt_boundary_wrfout_sampler(ID='',wrf_domain='d01',wrf_path='',trajectory_rds_filename='',bbox=None,rec_sample_vars=[],bnd_save_dir='./',trj_save_dir='./',overwrite_bnd=False,overwrite_trj=False):
    """
    Description:            Starting from a STILT trajectory RDS file, extracts the trajectories and locates the intersection with 
                            the boundary defined by bbox. Finds the wrfout file with the location/time of the boundary intersection
                            and extracts its CO2 and CH4 at the nearest model level with KDTree and writes the information to a 
                            file. Assumes hourly wrf output.
    Inputs:
        wrf_domain:         wrfout domain - "d01"
        wrf_path:           directory where wrfout files are stored
        trajectory_rds_filename full path of STILT particle trajectory file
        bbox                bounding box for domain - used in locate_trajectory_boundary_points
        rec_sample_vars     which variables to sample at receptor location from wrfout files
        bnd_save_dir        where to save boundary sample files
        trj_save_dir        where to save trajectory files
        overwrite_*         Boolean flag to overwrite existing bnd/trj files
                            well as any receptor info (e.g, pressure) that needs to be returned
    Returns:                filename of successfully created boundary point file (HDF5)

    """

    trj_fname = trj_save_dir+ID+'_traj.h5'
#---------
#   First find the locations of the boundary points 
#   Write from scratch:
    if overwrite_trj:
        return_dict = locate_trajectory_boundary_points(ID=ID,trajectory_rds_filename=trajectory_rds_filename,
                bbox=bbox,save_dir=trj_save_dir,write_files=True)

        bnd_loc_vars = return_dict['bnd_loc_vars']
        receptor_loc_vars = return_dict['receptor_loc_vars']
#   Otherwise look for existing files
    else:
        try:
            with File(trj_fname,'r') as f_in:
                obs_lat = f_in.attrs['obs_lat']
                obs_lon = f_in.attrs['obs_lon']
                obs_alt = f_in.attrs['obs_alt']
                obs_t = dt.datetime.strptime(f_in.attrs['obs_time'],'%Y-%m-%d %H:%M:%S UTC')#'2023-07-26 16:05:02 UTC')
                rec = {}
                rec['lat'] = f_in['particle_latitude'][:]
                rec['lon'] = f_in['particle_longitude'][:]
                rec['zagl'] = f_in['particle_altitude'][:]
                rec['last_ind'] = f_in['boundary_index'][:]
                rec['t'] = f_in['particle_time'][:]
            n_traj = rec['lat'].shape[0]
            inds = rec['last_ind'][:]

            receptor_loc_vars = {'lat':[obs_lat],'lon':[obs_lon],
                                            'datetime':[obs_t],
                                            'levs':[obs_alt]}
            bnd_loc_vars = {}
            bnd_loc_vars['bnd_lon'] = np.array([rec['lon'][i,inds[i]].mean() for i in range(n_traj)])
            bnd_loc_vars['bnd_lat'] = np.array([rec['lat'][i,inds[i]].mean() for i in range(n_traj)])
            bnd_loc_vars['bnd_zagl'] = np.array([rec['zagl'][i,inds[i]].mean() for i in range(n_traj)])
            bnd_loc_vars['t'] = np.array([rec['t'][i,inds[i]].mean() for i in range(n_traj)]) #time in seconds since release
            bnd_loc_vars['bnd_t'] = (obs_t-dt.datetime(1970,1,1)).total_seconds() + bnd_loc_vars['t'] # Seconds since 1970,1,1
    # if the boundary point files don't exist, create them
        except FileNotFoundError:
            return_dict = locate_trajectory_boundary_points(ID=ID,trajectory_rds_filename=trajectory_rds_filename,
                    bbox=bbox,save_dir=trj_save_dir,write_files=True)
            bnd_loc_vars = return_dict['bnd_loc_vars']
            receptor_loc_vars = return_dict['receptor_loc_vars']
    # Initialize the variables we want to sample at the receptor location
    for v in rec_sample_vars:
        receptor_loc_vars[v] = None

#--------
#   Now create the boundary point sample directories
    wrf_bnd_fname = ID+'_wrf'+wrf_domain+'.h5'
    os.makedirs(bnd_save_dir,exist_ok=True)
    if os.path.exists(bnd_save_dir+wrf_bnd_fname):
        if overwrite_bnd==True:
            os.remove(bnd_save_dir+wrf_bnd_fname)
        else:
            print('Boundary GHG file already exists, skipping.')
            return

    alt = bnd_loc_vars['bnd_zagl'][:]
    lat = bnd_loc_vars['bnd_lat'][:]
    lon = bnd_loc_vars['bnd_lon'][:]
    t = bnd_loc_vars['bnd_t'][:]
    obs_t = receptor_loc_vars['datetime']#dt.datetime(1970,1,1) + dt.timedelta(seconds=int(receptor_loc_vars['obs_t']))

    #==================================================
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

    # Get the met variables (or whatever) from the wrfout files at the receptor location and time
    if len(receptor_loc_vars.keys()) > 0:
        receptor_loc_vars = sample_wrfout_profile(wrf_domain=wrf_domain,wrf_path=wrf_path,var_dict=receptor_loc_vars)
    bc['receptor_loc_vars'] = receptor_loc_vars

    if os.path.exists(bnd_save_dir+wrf_bnd_fname): os.remove(bnd_save_dir+wrf_bnd_fname)

    with File(bnd_save_dir+wrf_bnd_fname,'w') as loc_f:
        g = loc_f.create_group('boundary')
        g.create_dataset('part_lat',data=lat[:])
        g['part_lat'].attrs['description'] = 'STILT particle latitude'
        g.create_dataset('part_lon',data=lon[:])
        g['part_lon'].attrs['description'] = 'STILT particle longitude'
        g.create_dataset('part_z',data=alt[:])
        g['part_z'].attrs['description'] = 'STILT particle altitude above the surface'
        g.create_dataset('part_t',data=t[:])
        g['part_t'].attrs['description'] = 'STILT particle altitude time (seconds since 1970-1-1 00:00:00 UTC)'
        g.create_dataset('wrf_bc_lat',data=bc['lat'][:])
        g['wrf_bc_lat'].attrs['description'] = 'NN WRF latitude'
        g.create_dataset('wrf_bc_lon',data=bc['lon'][:])
        g['wrf_bc_lon'].attrs['description'] = 'NN WRF longitude'
        g.create_dataset('wrf_bc_z',data=bc['z'][:])
        g['wrf_bc_z'].attrs['description'] = 'NN WRF altitude'
        g['wrf_bc_time'] = bc['t'][:]
        g['wrf_bc_time'].attrs['description'] = 'NN WRF time'
        g.create_dataset('wrf_bc_co2',data=bc['co2'][:])
        g['wrf_bc_co2'].attrs['description'] = 'NN WRF CO2 (ppm)'
        g.create_dataset('wrf_bc_ch4',data=bc['ch4'][:])
        g['wrf_bc_ch4'].attrs['description'] = 'NN WRF CH4 (ppm)'
        
        g = loc_f.create_group('receptor')
        for v in list(receptor_loc_vars.keys()):
            if v == 'datetime':
                out_v = []
                for d in receptor_loc_vars['datetime']:
                    out_v.append((d-dt.datetime(1970,1,1)).total_seconds())
                g.create_dataset('time',data=out_v)
            else:
                g.create_dataset(v,data=receptor_loc_vars[v])
        loc_f.close()
    return bnd_save_dir+wrf_bnd_fname

def compile_bnd_files(ID=None,wrf_domain='d01',profile_vars={},point_vars={},save_dir='./'):
    """
     ID 20230726_F1_05757
     vector vars are vectors like pressure, temp, water vapor, co2, ch4
     point vars are numbers like psfc, xco2, xch4, latitude, longitude, time
    """

    ob = ID.split('_')[-1]
    ob_files = sorted(glob.glob(f'{save_dir}/{ob:05d}/*{wrf_domain}.h5'))
    levs = sorted([fi.split('_')[3] for fi in ob_files])

    for v in list(boundary_vars.keys()):
        profile_vars[v]['values'] = np.zeros(len(levs))
    for iz,z in enumerate(levs):
        ind = np.where([fi.split('_')[3] == z for fi in ob_files])[0]
        f = File(ob_files[ind])
        for v in list(profile_vars.keys()):
            profile_vars[v][iz] = f[profile_vars[v]['bnd_group']][v]
    for v in list(point_vars.keys()):
        point_vars[v] = f[point_vars[v]['bnd_group']][v]

    return point_vars,profile_vars

def run_function_in_parallel(fun,args_list):
    """
    Description:        Generic function that runs any function over a set of CPUs -
                        note that this uses concurrent.futures.PoolProcessExecutor due
                        to multithreading issues with rpy2
    Inputs:
        fun             Python function
        args_list       List of dictionaries with keys set to match the input variable
                        argument names for the function fun. Note: each dictionary must have
                        a key called ID with unique value to be identifiable in the results dictionary.
    Returns:
        results_dict    Concatenated results of the function from collection of args_list
                        indexed by the ID key in the args_list list of dictionaries. Each entry
                        has a "result" entry that is the return value of the function, a "warnings" 
                        entry, and if the function does not conclude successfully, an error entry 
                        consisting of the Exception.
    """
    result_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to be executed in parallel with named arguments
        future_to_args = {executor.submit(fun, **args): args for args in args_list}

        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_args):
            args_future = future_to_args[future]
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = future.result()
                    result_dict[args_future['ID']] = {
                            "result": result,
                            "warnings": [str(warn.message) for warn in w],
                            "error": None
                            }
                print(f"Function with arg ID {args_future['ID']} finished successfully!")
            except Exception as exc:
                print(f"Function with arg ID {args_future['ID']} generated an exception: {exc}")
                result_dict[args_future['ID']] = {
                            "result": None,
                            "warnings": [],
                            "error": f"Generated an exception: {exc}"
                            }
    return result_dict
