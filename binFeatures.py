import numpy as np; import pandas as pd
import os; import time
import h5py; import imageio
from myutils import parallel_chunker, grab_files, limit_step
from lagrangian_processing import read_LagData, declare_bin_array, translate_part
from natsort import os_sorted
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD   # MPI stuff
rank = comm.Get_rank()  
size = comm.Get_size()

tic=time.time() # time on

best_features=['Plume area (um2)', 'Plume major axis (um)', 'Plume mean intensity',
                'Plume min intensity', 'Plume minor axis (um)', 'Spatter mean area',
                'Spatter mean mean intensity', 'Spatter number', 'Spatter total area',
                'Spot area (um2)', 'Spot major axis (um)', 'Spot mean intensity',
                'Spot minor axis (um)', 'X', 'Y','Z']   # define the feature that we want to keep from metrics, can be adjusted
stat_params=['mean', 'max', 'mse', 'raw']   # name of the statistical parameters we want to calculate for the bins, raw = collect all data points
BIN_RAW=False   # a switch whether to include the raw data in the bins or not

data_dir=os.path.normpath("E:\\Rich\\Data\\ID_v3\\METRICS")     # path to layer metrics dataframes 
part_name="METRICS_ONLY_2mm"     # name of part, for writing out, don't need file ext
out_dir=os.path.join(data_dir+'\\BINNED')   # where we are going to write to
if rank==0:
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)       # here we are making the folder if it doesn't already exist
        
proc_files=grab_files(data_dir, ext='.csv')     # create a list of all the layer file objects
proc_files=os_sorted(proc_files)    # sort into numerical order, not 1, 100, 101 etc. 

#** NEEDS CALIBRATING **
# load in dummy layer file to calibrate some constants for the binning, must select the layer with the largest xy footprint
# here to ensure the full space gets binned
df=pd.read_csv(proc_files[1])  

bin_XYsize=2.0; bin_Zsize=2.0; layer_height=0.05; build_height=len(proc_files)*layer_height; PPCM=7000  # set constants and bin size
# bin size & layer height expressed in mm
#coords=[[-21628.277709730297, 2878.4381222563497], [-9992.408685006729, -17271.870686560305]]
comm.Barrier()      # all procs wait here

if rank==0:
    array_x, array_y, array_z, x_coord,y_coord, ref_x, ref_y = declare_bin_array(df, bin_XYsize, bin_Zsize, 
        layer_height, build_height)     # will plot the dummy layer we selected earlier and we then click through the corners of the area to be binned
                                        # first click on the top left (NW) corner, second click bottom right (SE).
    config=[array_x, array_y, array_z, x_coord, y_coord, ref_x, ref_y]  # save some config params in a list
else:
    config=None
config = comm.bcast(config, root=0)     # Proc 1 will broadcast the config params to the other workers
comm.Barrier()      # again we wait

sub_dict={}
for param in stat_params:
    sub_dict[param]={}  # build first nested layer of the sub dictionary for each worker to collect the binned data

for param in stat_params:
    for key in best_features:      # nest the second layer, all the features underneath the statistical parameter
        if param=='raw' and BIN_RAW==True:
            sub_dict[param][key]=np.empty((math.ceil(config[2]/size), config[1], config[0], PPCM*(bin_XYsize**3)),
                                        dtype='float64')
            sub_dict[param][key][:]=np.nan  # preallocate oversized 4D nan array to read in the raw data vector at each bin.
        else:                               # length of this nan array was figured out empirically, approx. 7000 for 2 mm bin
            sub_dict[param][key]=np.zeros((math.ceil(config[2]/size), config[1], config[0]),
                                        dtype='float64')       # only 3D needed for statistical parameters since they are scalar
                                      
#if rank==0:
#    ID=np.zeros((config[1], config[0]), dtype='uint8')
layers_per_voxel=math.ceil(bin_Zsize/layer_height)

voxel_layer=0
#if layers_per_voxel==1:
#    for start in parallel_chunker(len(proc_files), layers_per_voxel, size, rank):
#        file=proc_files[start]    
#        df=read_LagData(file, df)   

for start, end in parallel_chunker(len(proc_files), layers_per_voxel, size, rank):      # the big loop through the bin layers in z
    df=pd.DataFrame()
    for file in proc_files[start:end]:
        df=read_LagData(file, df)       # daisy-chain together all layer dataframes relevant to bin layer

    #df['X'], df['Y'] = rotate_xy_pos(df['X'].to_numpy(), df['Y'].to_numpy(), 0)        # used if we needed to rotate the scan scene
    df=df[df['Spot area (um2)']>5]   #and bits in the middle where we are not melting i.e. no spot in these bits. 
    df=df[df['Plume area (um2)']>2]
    df.fillna(0, inplace=True)          # some cleaning to remove anything with no spot, no plume and fill nans
    
    df=df[(config[3][0]<df.X) & (df.X<config[3][1]) & (config[4][0]<df.Y) & (df.Y<config[4][1])]    # crop data to include only selected part

    if len(df)==0:  # here our bin layer contains no data, will happen when melting of selected part finished but data continues for taller part. Will exit loop
        print('out of range')
        break           

    df.X, df.Y,_,_= translate_part(df.X, df.Y, config[5], config[6])    # translate onto local coordinates (bottom left-most corner of part, from a bird's eye perspective = (0,0))
    
    df.X, df.Y= df.X/1010, df.Y/1010    # scale into mm
        
    for x in range(0, config[0]):
        lower_x=np.round(x*bin_XYsize, 2)
        upper_x=(x*bin_XYsize)+bin_XYsize
        
        for y in range(0, config[1]):   # loop through the bin layer in x, y 
            lower_y=np.round(y*bin_XYsize, 2)
            upper_y=np.round((y*bin_XYsize)+bin_XYsize, 2)      # some crap to do with floats and rounding in python. Potentially not needed.

            data=df[(lower_x<=df.X)&(df.X<=upper_x)&(lower_y<=df.Y)&(df.Y<=upper_y)] #index out the data points for our bin
            #if rank==0 and start==0:
                #if len(data)>10:
                    #ID[y,x]=max(data['ID'])+1 #bin the burnt in data
                    #print("binning burnt-in ID info")
            vector_size=len(data)       # calculate number of data points in bin
            for i, prop in enumerate(best_features):
                if vector_size > 50:        # bin the mean, max, variance and raw data (if selected) if there are more than x data points in bin
                    sub_dict['mean'][prop][voxel_layer,y,x]=np.mean(data[prop])     
                    sub_dict['max'][prop][voxel_layer,y,x]=max(data[prop])
                    sub_dict['mse'][prop][voxel_layer,y,x]=np.var(data[prop])
                    if BIN_RAW==True:
                        sub_dict['raw'][prop][voxel_layer,y,x,:vector_size]=data[prop]
                    
    voxel_layer=voxel_layer+1   
    print('Proc {0} Completed voxel layer {1}'.format(rank, voxel_layer), flush=True)

with h5py.File(os.path.join(out_dir, ('_{0}_'.format(rank)+part_name+'.h5')), 'w') as f:    # each process writes out its bit of the binned array
    for param in stat_params:
        for key in best_features:
            f.create_dataset('{0}-{1}'.format(param, key), shape=sub_dict[param][key].shape, dtype='float64',
                data=sub_dict[param][key])
            
comm.Barrier()      # wait til all are done, only proc 1 proceeds now to reassemble to sub-arrays

if rank==0:
    
    main_dict={}
    for param in stat_params:
        main_dict[param]={}
    
    for param in stat_params:       # building the master data structure
        for key in best_features:
            if param=='raw' and BIN_RAW==True:
                main_dict[param][key]=np.empty((config[2], config[1], config[0], PPCM*(bin_XYsize**3)),
                                        dtype='float64')
                main_dict[param][key][:]=np.nan
            else:
                main_dict[param][key]=np.zeros((config[2], config[1], config[0]), dtype='float64')
            
            z=0
            for i in range(math.ceil(config[2]/size)):  # loop in z through master bin layers
                for j in range(size):       # loop through the procs
                    with h5py.File(os.path.join(out_dir, ('_{0}_'.format(j)+part_name+'.h5')),'r') as f:
                        main_dict[param][key][z]=np.array(f['{0}-{1}'.format(param,key)][i])        
                        # pulled out the binned slice from sub-file into master array, for each feature and statistical parameter
                        
                        if np.all(np.array(f['{0}-{1}'.format(param,key)][i])==0):
                            print("blank slice", flush=True)
                            continue        # sub file contains a blank slice, move on. Can occur when some procs have more layers to complete than others 
                        
                        #main_dict['layer'][z]=f['layer']
                    #os.remove(os.path.join(out_dir, ('_{0}_'.format(j)+part_name+'.h5')))
                    #print("removing temp files", flush=True)
                    z=z+1   
                    if z==config[2]:
                        break
                    
    out_keys=[i for i in main_dict.keys() if 'id' not in i]     
    with h5py.File(os.path.join(out_dir, part_name+'.h5'), 'w') as f:
        #f.create_dataset('ID', shape=ID.shape, dtype='uint8', data=ID)
        for param in out_keys:
            for key in main_dict[param].keys():
                if key!='layer':
                    f.create_dataset("{0}-{1}".format(param, key), shape=main_dict[param][key].shape,
                                     dtype='float64', data=main_dict[param][key])
                    if param!='raw':
                        #imageio.volwrite(os.path.join(out_dir, (part_name+'_{0}-{1}.tiff'.format(key, param))), 
                        #        main_dict[param][key].astype('uint32'), format='TIFF')
                    
    
    toc=time.time()
    for i in range(size):
        os.remove(os.path.join(out_dir, ('_{0}_'.format(i)+part_name+'.h5')))
    print("deleted temp files", flush=True)
    print('finished in {0} secs'.format(toc-tic), flush=True)
