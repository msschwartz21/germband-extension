import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from czifile import CziFile
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.measure import grid_points_in_poly

import tqdm

from scipy.interpolate import RectBivariateSpline

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
        try:
            L.append(np.array(f.get('t'+str(t)).get(dataset)))
        except:
            print('No data at time',t)
            
    # Close h5 file
    f.close()
        
    # Join all timepoints into a single numpy array to return
    return(np.stack(L))

def write_hyperstack(hst,fpath,dataset='channel1'):
    '''
    Write an h5 file in the same format as was read in
    
    Parameters
    ----------
    hst : np.array
        Array with dimensions txy
    fpath : str
        Complete path to output file with '.h5' 
    dataset : str, optional
        Specifies name of the dataset in h5 file
    '''
    
    # Open new h5py file to add data to
    f = h5py.File(fpath,'w')

    # Save each timepoint to a group/dataset in h5 file
    for t in range(hst.shape[0]):
        f.create_dataset('t'+str(t)+'/'+dataset, data=hst[t])

    # close file
    f.close()

def imshow(img,figsize=(10,8)):
    '''
    Show image using matplotlib and including colorbar
    '''
    
    fig,ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(img)
    plt.colorbar(cax)
    
class MaskEmbryo():
    '''
    Fit an ellipse to an embryo and calculate mask
    '''

    def __init__(self,points):
        '''
        Calculate a first try ellipse using default parameters

        Parameters
        ----------
        points : pd.DataFrame
            Contains the columns x and y with 2 rows

        Returns
        -------
        self.ell, self.rell, self.fell
        '''

        self.df = points
        self.calc_start_ell()
        self.calc_rotation()
        self.shift_to_center()
    
    def calc_ellipse(self,center_x,center_y,radius_x,radius_y):
        '''
        Calculate a parametrized ellipse based on input values
        
        Parameters
        ----------
        center_x : float
            Center point of the ellipse in x dimension
        center_y : float
            Center point of the ellipse in y dimension
        radius_x : float
            Radius of ellipse in x dimension
        radius_y : float
            Radius of ellipse in y dimension

        Returns
        -------
        Ellipse in a 400x2 array
        '''

        # Values to calculate the ellipse over 
        # using parametrized trigonometric fxns
        s = np.linspace(0, 2*np.pi, 400)

        # Calculate the position of the ellipse as a function of s
        x = center_x + radius_x*np.cos(s)
        y = center_y + radius_y*np.sin(s)
        init = np.array([x, y]).T

        return(init)

    def calc_start_ell(self,scale=1.5,yradius=300,df=None):
        '''
        Customize the fit of an ellipse to an embryo based on the selected endpoints

        Parameters
        ----------
        scale : float, optional
            Typically greater than 1 to extend the length between the two points beyond the ends of the embryo
        yradius : int, optional
            Y radius for initial ellipse, default=300
        df : pd.DataFrame, optional
            Contains the columns x and y with 2 rows
        
        Returns
        -------
        self.ell : array
            Array of shape 400x2 that contains position of custom ellipse
        '''

        # Assign global variables if not specified
        if df == None:
            df = self.df

        # Calculate the length of the embryo based on two endpoints
        l = np.sqrt((df.iloc[1].x - df.iloc[0].x)**2 + 
                    (df.iloc[1].y - df.iloc[0].y)**2)

        # Divide l by 2 and scale by scale factor
        self.radius = (l/2)*scale

        # Calculate ellipse at 0,0
        self.ell = self.calc_ellipse(0,0,self.radius,yradius)

    def calc_rotation(self,ell=None,df=None):
        '''
        Calculate angle of rotation and rotation matrix using -angle

        Parameters
        ----------
        ell : np.array, optional
            Ellipse array
        df : pd.DataFrame, optional
            Contains the columns x and y with 2 rows
        
        Returns
        -------
        self.rell
        '''

        # Assign global variables if not specified
        if ell == None:
            ell = self.ell
        if df == None:
            df = self.df

        # Calculate rotation angle using arctan with two end points
        theta = -np.arctan2(df.iloc[0].y-df.iloc[1].y, 
                           df.iloc[0].x-df.iloc[1].x)

        # Calculate rotation matrix based on theta and apply to ell
        R = np.array([[np.cos(theta),-np.sin(theta)],
                 [np.sin(theta),np.cos(theta)]])
        self.rell = np.dot(ell,R)

    def shift_to_center(self,rell=None,df=None):
        '''
        Shift ellipse that started at (0,0) to the center of the embryo

        Parameters
        ----------
        rell : np.array, optional
            Ellipse array
        df : pd.DataFrame, optional
            Contains the columns x and y with 2 rows
        
        Returns
        -------
        self.fell
        '''

        # Assign global variables if not specified
        if rell == None:
            rell = self.rell
        if df == None:
            df = self.df

        # Calculate the center embryo point based on endpoints
        centerx = np.abs(df.iloc[0].x - df.iloc[1].x)/2
        centery = np.abs(df.iloc[0].y - df.iloc[1].y)/2

        # Calculate shift from origin to embryo center
        yshift = df.y.min()
        xshift = df.x.min()
        centerx = centerx + xshift
        centery = centery + yshift

        # Shift rotated ellipse to the center
        self.fell = np.zeros(rell.shape)
        self.fell[:,0] = rell[:,0]+centerx
        self.fell[:,1] = rell[:,1]+centery

        return(self.fell)


    def contour_embryo(self,img,init=None,sigma=3):
        '''
        Fit a contour to the embryo to separate the background

        Parameters
        ----------
        img : 2D np.array
            2D image from a single timepoint to mask
        init : 400x2 ellipse array, optional
            Starting ellipse array that is bigger than the embryo
        sigma : int, optional 
            Kernel size for the Gaussian smoothing step

        Returns
        -------
        Masked image where all background points = 0
        '''

        # Assign global variables if not specified
        if init == None:
            init = self.fell

        # Fit contour based on starting ellipse
        snake = active_contour(gaussian(img, sigma),
                           init, alpha=0.015, beta=10, gamma=0.001)

        # Create boolean mask based on contour
        mask = grid_points_in_poly(img.shape, snake).T

        return(mask)
    
    def mask_image(self,img,mask):
        '''
        Apply mask to image and return with background = 0
        
        Parameters
        ----------
        img : 2D np.array
            2D image from a single timepoint to mask
        mask : 2D np.array
            2D boolean array containing mask
        '''
        
        # Apply mask to image and set background to 0
        img[~mask] = 0
        
        return(img)

class CziImport():
    '''
    Defines a class to wrap the czifile object
    Identifies data contained in each dimension
    Helps user extract desired data from multidimensional array
    '''
    
    def __init__(self,fpath,summary=True):
        '''
        Read in file using czifile
        
        Parameters
        ----------
            fpath : str
                Complete or relative file path to czi file
        '''
        
        with CziFile(fpath) as czi:
            self.raw_im = czi.asarray()
        
        if summary:
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
        
        print(self.raw_im.shape)
              
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
        
        # Check the length of first axis and swap dimensions if greater than a few
        if self.data.shape[0] > 4:
            self.data = np.swapaxes(self.data,0,1)
        
def tidy_vector_data(name):
    '''
    Tidys csv files exported from matlab OpticalFlow
    
    Parameters
    ----------
    name : str
        String specifying the name passed to OpticalFlowOutput 
        Can include complete or partial path to file
        
    Returns
    -------
    pd.DataFrame
        Dataframe containing the following columns:
        frame, x, y, vx, vy
    '''
    
    # Read in csv files with x and y positions
    x = pd.read_csv(name+'_X.csv', header=None)
    y = pd.read_csv(name+'_Y.csv', header=None)
    
    # Concatenate x and y which contain the positions of vectors
    # Rename default column name to x or y
    # Create position ID column for joining
    xy = pd.concat([x.rename(columns={0:'x'}),
                   y.rename(columns={0:'y'})
                   ],axis=1).reset_index(
                  ).rename(columns={'index':'position ID'})
    
    # Define column subsets for melt function
    id_vars = ['position ID','x','y']
    value_vars = np.arange(0,166)
    
    # Read vx and vy
    # Join with xy to add position values
    vx = pd.read_csv(name+'_Vx.csv', header=None
                    ).join(xy
                    ).melt(id_vars=id_vars,
                         value_vars=value_vars,
                         var_name='frame',
                         value_name='vx')
    vy = pd.read_csv(name+'_Vy.csv', header=None
                    ).join(xy
                    ).melt(id_vars=id_vars,
                         value_vars=value_vars,
                         var_name='frame',
                         value_name='vy')
    
    vectors = vx.merge(vy)
    
    return(vectors)

def reshape_vector_data(df):
    '''
    Convert dataframe structure into a set of meshgrid arrays
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns x,y,frame,vx,vy
        
    Returns
    -------
    tt,xx,yy,vx,vy
        Set of arrays of shape (len(T),len(X),len(Y))
    '''
    
    # Extract unique values for txy
    T = df['frame'].unique()
    X = df['x'].unique()
    Y = df['y'].unique()
    
    # Create meshgrid using 'ij' indexing to get shape txy
    tt,xx,yy = np.meshgrid(T,X,Y,indexing='ij')
    
    # Create hierarchical index 
    dfh = df.set_index(['frame','x','y'])
    
    # Reshape vx and vy values into numpy array
    vx = dfh['vx'].values.reshape((T.shape[0],X.shape[0],Y.shape[0]))
    vy = dfh['vy'].values.reshape((T.shape[0],X.shape[0],Y.shape[0]))
    
    return(tt,xx,yy,vx,vy)

def calc_flow_path(xval,yval,vx,vy,x0,y0):
    
    # Initialize position list with start value
    xpos = [x0]
    ypos = [y0]
    
    for t in tqdm.tqdm(range(1,vx.shape[0])):
        
        # Interpolate to find change in x and y
        dx = RectBivariateSpline(xval,yval,vx[t]).ev(xpos[t-1],ypos[t-1])
        dy = RectBivariateSpline(xval,yval,vy[t]).ev(xpos[t-1],ypos[t-1])

        # Update position arrays
        xpos.append(xpos[t-1]+dx)
        ypos.append(ypos[t-1]+dy)
        
    return(np.array([xpos,ypos]))