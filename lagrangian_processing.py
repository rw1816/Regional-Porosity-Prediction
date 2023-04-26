import numpy as np
import pandas as pd
#from numba import jit
import h5py
from myutils import get_crop_coords
import matplotlib.pyplot as plt
import matplotlib
import math
import time

#@jit(nopython=True, parallel=True, fastmath=True)
def rotate_xy_pos(x, y, angle, centre=(0,0)):
    
    """
    Rotate xy position matrix by a given angle around a given origin to make
    the part square/ parallel with xy axes.

    The angle should be given in degrees.
    """
    angle = np.radians(angle)
    ox, oy = centre[0], centre[1]

    x_square = ox - np.cos(angle) * (x - ox) - np.sin(angle) * (y - oy)
    y_square = oy - np.sin(angle) * (x - ox) + np.cos(angle) * (y - oy)
   
    return x_square, y_square


def translate_part(x, y, ref_x=0, ref_y=0, target=(0,0)):

    """
    Translate part xy position from machine coordinates onto a part coordinate
    system, with the bottom left hand corner as (0,0). Specify optional target
    parameter to map bottom left to different values.
    
    Assumes the rotation step above has been applied and the part is oriented
    as per FE model view.
    
    """
    if not ref_x:
        
        ref_x=min(x)
        ref_y=min(y)
        print("setting reference coords ...")
        
    x_new=x-ref_x+target[0]+1
    y_new=y-ref_y+target[1]+1
        
    return x_new, y_new, ref_x, ref_y

def drop_borders(df):
    
        min_true=np.array(np.where(df.power_act<15)).flatten()
        min_true=np.append(min_true, len(df))
        border_info=np.array([i for i in enumerate(np.diff(min_true)) if i[1]>2500])
                #grabs start point and length of borders
        border_info[:,0]=min_true[border_info[:,0]]
        make_bool=np.zeros(len(df))
        
        for i,j in border_info:
            make_bool[i:i+j]=1
        df=df[~make_bool.astype('bool')]
        
        return df
    
def declare_voxel_space(layerfiles, bin_size, build_height, coords=None):
    
    for filename in layerfiles[0:1]:
        with h5py.File(filename, 'r') as f:
            data=f.get('data')
            data=np.array(data)
            df=pd.DataFrame(np.int16(data), columns=('raw_x_req','corr_x_req', 'raw_y_req', 
                                                     'corr_y_req','z_req', '6','raw_x_act','corr_x_act','raw_y_act', 
                                                     'corr_y_act','z_act', '12','power_req','power_act', 'power_actCOPY',
                                                     'PD1', 'PD2','18','19','end1','end2','end3'))
    if not coords:
        coords=[]
        coords=get_crop_coords(df.corr_x_req, df.corr_y_req)
    
    x_min=min(coords[0][0], coords[1][0])
    x_max=max(coords[0][0], coords[1][0])
    y_min=min(coords[0][1], coords[1][1])
    y_max=max(coords[0][1], coords[1][1])
    df=df[(x_min<df.corr_x_req) & (df.corr_x_req<x_max) & (y_min<df.corr_y_req) & (df.corr_y_req<y_max)]
    df=df[['corr_x_req','corr_y_req', 'power_req','power_act', 'PD1', 'PD2']]
    del(data)

    #rotate part to a front view parallel with xy axes
    df['corr_x_req'], df['corr_y_req'] = rotate_xy_pos(df.corr_x_req.to_numpy(), df.corr_y_req.to_numpy(), 201)
    df=df[df.index>(max(df.index)*0.075)] #drop the crap at the beginning where the mirrors are moving around
    df=df[df.PD1>300]   #and bits in the middle where we are not melting. very low PD (<100) value in these bits. 
    #translate part onto part coordinates, bottom left corner 0,0
    df['corr_x_req'], df['corr_y_req'], ref_x, ref_y = translate_part(df['corr_x_req'], df['corr_y_req'], 0)
    
    # convert to true scale (int16 --> mm)
    df['corr_x_req'], df['corr_y_req'] = df['corr_x_req']/200, df['corr_y_req']/200
    # now we can start binning
    #declare a blank voxel array to read into 
    #how many do we need?
    array_x=math.ceil(max(df.corr_x_req)/bin_size)
    array_y=math.ceil(max(df.corr_y_req)/bin_size)
    array_z=math.ceil(build_height/bin_size)
    
    x=(x_min, x_max)
    y=(y_min, y_max)
    return array_x, array_y, array_z, x, y, ref_x,ref_y

def read_LagData(file, df, props=0):
    
    if file.name[-4:] == '.csv':
        d_new=pd.read_csv(file)
        df=df.append(d_new)
        return df
    
    elif file.name[-3:]=='.h5':
         with h5py.File(file,'r') as f:
            data=np.array(f.get('data'), dtype='float64')        
         df=df.append(pd.DataFrame(data, columns=['time','power','x','y', *props], 
                                   dtype='float64'))
         return df
    else:
        print('File Format not recognised')
        
def declare_bin_array(df, bin_XYsize, bin_Zsize, layer_height, build_height, coords=None):
    
    if not coords:
        coords=get_crop_coords(df.X, df.Y)
        
    x_min=min(coords[0][0], coords[1][0])
    x_max=max(coords[0][0], coords[1][0])
    y_min=min(coords[0][1], coords[1][1])
    y_max=max(coords[0][1], coords[1][1])
    df=df[(x_min<df.X) & (df.X<x_max) & (y_min<df.Y) & (df.Y<y_max)]
    
    df=df[df['Spot area (um2)']>1]   #and bits in the middle where we are not melting. very low PD (<100) value in these bits. 
    df=df[df['Plume area (um2)']>1]
    df.dropna(0,)
    df.X, df.Y, ref_x, ref_y = translate_part(df['X'].to_numpy(), df['Y'].to_numpy())
    df.X, df.Y= df.X/1010, df.Y/1010
    
    array_x=math.ceil(max(df.X)/bin_XYsize)
    array_y=math.ceil(max(df.Y)/bin_XYsize)
    array_z=math.ceil((build_height/bin_Zsize))
    
    x_coord=(x_min, x_max)
    y_coord=(y_min, y_max)
    
    return array_x, array_y, array_z, x_coord, y_coord, ref_x,ref_y
