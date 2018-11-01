import h5py
import numpy as np
from czifile import CziFile

import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.measure import grid_points_in_poly

def read_hyperstack(fpath,dataset='channel1',tmax=166): 
    '''
    Read in an hdf5 hyperstack where each timepoint is a group
    containing two channels which are saved as datapoints
    '''
    
    # Open file object
    f = h5py.File(fpath,'r')
    
    # Retrieve each timepoint from file object
    L = []
    for t in range(tmax+1):
        L.append(np.array(f.get('t'+str(t)).get(dataset)))
        
    # Close h5 file
    f.close()
        
    # Join all timepoints into a single numpy array to return
    return(np.stack(L))

def write_hyperstack(fpath,dataset='channel1'):
    '''
    Write an h5 file in the same format as was read in
    '''
    
    # Open new h5py file to add data to
    f = h5py.File('../data/wt_gbe_20180110_mask.h5','w')

    # Save each timepoint to a group/dataset in h5 file
    for t in range(hst.shape[0]):
        f.create_dataset('t'+str(t)+'/channel1', data=hst[t])

    # close file
    f.close()

def imshow(img,figsize=(10,8)):
    '''
    Show image using matplotlib and including colorbar
    '''
    
    fig,ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(img)
    plt.colorbar(cax)
    
def calc_ellipse(center_x,center_y,radius_x,radius_y):
    '''
    Calculate a parametrized ellipse based on input values
    '''
    
    # Values to calculate the ellipse over using parametrized trigonometric fxns
    s = np.linspace(0, 2*np.pi, 400)
    
    # Calculate the position of the ellipse as a function of s
    x = center_x + radius_x*np.cos(s)
    y = center_y + radius_y*np.sin(s)
    init = np.array([x, y]).T
    
    return(init)

def contour_embryo(img,init):
    '''
    Fit a contour to the embryo to separate the background
    Returns a masked image where all background points = 0
    '''
    
    # Fit contour based on starting ellipse
    snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=10, gamma=0.001)
    
    # Create boolean mask based on contour
    mask = grid_points_in_poly(img.shape, snake).T
    
    # Apply mask to image and set background to 0
    img[~mask] = 0
    
    return(img)

class CziImport():
    '''
    Defines a class to wrap the czifile object
    Identifies data contained in each dimension
    Helps user extract desired data from multidimensional array
    '''
    
    def __init__(self,fpath):
        '''
        Read in file using czifile
        
        Parameters
        ----------
            fpath : str
                Complete or relative file path to czi file
        '''
        
        with CziFile(fpath) as czi:
            self.raw_im = czi.asarray()
            
        self.print_summary()
        self.squeeze_data()
                
    def print_summary(self):
        '''Prints a summary of data dimensions
        Assumes that the data is a standard brightfield timelapse collection, e.g. (?, roi, channel, time, z, x, y, ?)
        '''
        
        print('''
            There are {0} ROIs,
                       {1} channels and
                       {2} timepoints.
            '''.format(self.raw_im.shape[1],
                      self.raw_im.shape[2],
                      self.raw_im.shape[3]))
              
        print('''
            The 3D dimensions of the data are:
            {0} x {1} x {2} (zyx)
            '''.format(self.raw_im.shape[-4],
                      self.raw_im.shape[-3],
                      self.raw_im.shape[-2]))
        
    def squeeze_data(self):
        '''
        Uses np.squeeze to reduce dimenions of data according to input preference
        '''
        
        self.data = np.squeeze(self.raw_im)