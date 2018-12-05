
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import bebi103

from czifile import CziFile
import tifffile
import av

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
    
def calc_line(points):
    '''
    Given two points calculate the line between the two points
    
    Parameters
    -----------
    points : pd.DataFrame
        Dataframe containing the columns x, y, and 'level_1' (from df.reset_index())
        Each row should contain data for a single point
        
    Returns
    --------
    lines : pd.DataFrame
        Dataframe with columns that specify an equation for a line
    '''
    
    line = pd.DataFrame()
    
    # Specify columns for x1,y1,x2,y2
    line['x1'] = points[points['level_1']==0]['x']
    line['x2'] = points[points['level_1']==1]['x']
    line['y1'] = points[points['level_1']==0]['y']
    line['y2'] = points[points['level_1']==1]['y']
    
    # Calculate the slope
    line['dx'] = line['x2'] - line['x1']
    line['dy'] = line['y2'] - line['y1']
    line['m'] = line['dy']/line['dx']
    
    return(line)

def calc_embryo_theta(line):
    '''
    Given a line fit to each embryo, calculate the angle of rotation
    to align the embryo in the horizontal axis
    
    Parameters
    -----------
    line : pd.DataFrame
        Dataframe returned by :func:`calc_line` containing columns dy and dx
        
    Returns
    -------
    line : pd.DataFrame
        Input dataframe with an additional column theta in degrees
    '''
    
    line['theta'] = np.rad2deg(np.arctan2(line['dy'], line['dx']))
    return(line)
    
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

def calc_flow_path(xval,yval,vx,vy,x0,y0,dt,timer=True):
    '''
    Calculate the trajectory of a point through the vector field over time
    
    Parameters
    ----------
    xval : np.array
        A list of unique x values that define the meshgrid of xx
    yval : np.array
        A corresponding list of unique y values that define the meshgrid of yy
    vx : np.array
        Array of shape (time,len(xval),len(yval)) containing the x velocity component
    vy : np.array
        Array of shape (time,len(xval),len(yval)) containing y velocity component
    dt : float
        Duration of the time step between intervals
    timer : boolean, optional
        Default true uses the tqdm timer as an iterator
        
    Returns
    -------
    Array of shape (time,2) containing x and y position of trajectory over time
    
    '''
    
    # Initialize position list with start value
    xpos = [x0]
    ypos = [y0]
    
    if timer == True:
        iterator = tqdm.tqdm(range(1,vx.shape[0]))
    else:
        iterator = range(1,vx.shape[0])
    
    for t in iterator:
        
        # Interpolate to find change in x and y
        dx = RectBivariateSpline(xval,yval,dt*vx[t]
                                ).ev(xpos[t-1],ypos[t-1])
        dy = RectBivariateSpline(xval,yval,dt*vy[t]
                                ).ev(xpos[t-1],ypos[t-1])

        # Update position arrays
        xpos.append(xpos[t-1]+dx)
        ypos.append(ypos[t-1]+dy)
        
    return(np.array([xpos,ypos]))

class VectorField:
    '''
    Object to manage results and calculations from OpticalFlow.mat
    
    See `__init__` for more information
    '''
    
    def __init__(self,name):
        '''
        Initialize VectorField object by importing and transforming data
        
        Parameters
        ----------
        name : str
            String specifying the name passed to OpticalFlowOutput 
            Can include complete or partial path to file
            
        Attributes
        ----------
        name : str
            Based on the parameter name
        df : pd.DataFrame
            Dataframe of vectors produced by `tidy_vector_data`
        tt : np.array
            Meshgrid for t dimension
        xx : np.array
            Meshgrid for x dimension
        yy : np.array
            Meshgrid for y dimension
        vx : np.array
            X component of the velocity vector
        vy : np.array
            Y component of the velocity vector
        xval : np.array
            Sorted array of unique x values in xx meshgrid
        yval : np.array
            Sorted array of unique y values in yy meshgrid
        tval : np.array
            Sorted array of unique t values in tt meshgrid
        starts : pd.DataFrame
            Dataframe initialized to contain starter points
        '''
        
        self.name = name
        
        # Import vector data as dataframe
        self.df = tidy_vector_data(self.name)
        
        # Transform vector dataframe into arrays
        self.tt,self.xx,self.yy,self.vx,self.vy = reshape_vector_data(self.df)
        
        # Extract unique x and y positions
        self.xval = np.unique(self.xx)
        self.yval = np.unique(self.yy)
        self.tval = np.unique(self.tt)
        
        # Initialize start point dataframe
        self.starts = pd.DataFrame()
        
    def add_image_data(self,impath):
        '''
        Imports a 3D (txy) dataset (czi or tiff) that matches the vector data
        
        Parameters
        ----------
        impath : str
            Complete or relative path to image file
            Accepts either tif or czi file types
            
        Attributes
        ----------
        img : np.array
            3D array of image data
        '''
        
        # Determine file type for import
        if impath[-3:] == 'czi':
            self.img = CziImport(impath,summary=False).data
        elif 'tif' in impath[-4:]:
            self.img = tifffile.imread(impath)
        else:
            print('Image files must be czi or tif')
            
    def pick_start_points(self,notebook_url='localhost:8888'):
        '''
        Launches interactive bokeh plot to record user clicks
        
        Parameters
        ----------
        notebook_url : str, optional
            Default 'localhost:8888', specifies jupyterlab url for
            interactive plotting
        
        Returns
        -------
        p 
            Plotting object for bokeh plot
        '''
        
        # Record start clicks
        p = bebi103.viz.record_clicks(self.img[0],
                                      notebook_url=notebook_url,
                                      flip=False)
        
        return(p)
    
    def save_start_points(self,p):
        '''
        Uses the `to_df` method of the plotting object generated
        by `pick_start_points` to generate dataframe of click points
        
        Parameters
        ----------
        p : object
            Generated by `pick_start_points` after clicks have 
            been recorded
        
        Attributes
        ----------
        starts : pd.DataFrame
            Appends new recorded clicks to `starts` dataframe
        '''
        
        # Add to starts dataframe
        self.starts = self.starts.append(p.to_df())
        
    def initialize_interpolation(self,timer=True):
        '''
        Calculates interpolation of vx and vy for each timepoint
        Uses scipy.interpolate.RectBivariateSpline for optimal speed
        on meshgrid data
        
        Parameters
        ----------
        timer : boolean, optional
            Default = True, activates tqdm progress timer
            
        Attributes
        ----------
        Ldx : list
            List of dx interpolation objects for each t
        Ldy : list
            List of dy interpolation objects for each t
        interp_init : boolean
            Set to True after completion of interpolation for loop
        '''
        
        # Store interpolation object over time
        self.Ldx = []
        self.Ldy = []
        
        # Record interpolation initializationa as false
        self.interp_init = False
        
        # Set iterator with or without tqdm
        # Includes zeroth timepoint where all vx and vy = 0
        if timer == True:
            iterator = tqdm.tqdm(self.tval)
        else:
            iterator = self.tval

        for t in iterator:

            # Interpolate to find change in x and y
            dx = RectBivariateSpline(self.xval,self.yval,
                                     self.vx[t])
            dy = RectBivariateSpline(self.xval,self.yval,
                                     self.vy[t])
            
            # Save iterator to list
            self.Ldx.append(dx)
            self.Ldy.append(dy)
            
        # Set interpolation initialization value to True
        self.interp_init = True
        
    def calc_track(self,x0,y0,dt,tmin=0):
        '''
        Calculate the trajectory of a single point through space and time
        
        Parameters
        ----------
        x0 : float
            X position of the starting point
        y0 : float
            Y position of the starting point
        dt : float
            Duration of time step
        trange : list or np.array
            Range of t values to iterate over for interpolation
            
        Returns
        -------
        track : np.array
            Array of dimension number_t_steps x 2
        '''
        
        # Check if interpolation has been initialized
        if hasattr(self,'interp_init') and (self.interp_init==True):
            # Continue with function without problem
            pass
        else:
            self.initialize_interpolation()
            
        trange = range(tmin,np.max(self.tval))
            
        # Initialize position list with start value
        xpos = [x0]*(tmin+1)
        ypos = [y0]*(tmin+1)
        
        for t in trange:
            
            # Calculate dx and dy from iterators
            dx = self.Ldx[t].ev(xpos[t],ypos[t])
            dy = self.Ldy[t].ev(xpos[t],ypos[t])
            
            # Update position arrays
            # Multiply velocity vector by time to get distance
            xpos.append(xpos[t] + dx*dt)
            ypos.append(ypos[t] + dy*dt)
            
        return(np.array([xpos,ypos]))
    
    def calc_track_set(self,starts,dt,name='',timer=True,tmin=0):
        '''
        Calculate trajectories for a set of points using a constant dt
        
        Parameters
        ----------
        starts : pd.DataFrame
            Dataframe with columns x and y containing one point per row
        dt : float
            Duration of time step
        name : str, optional
            Default, '', encodes notes for a set of points 
        timer : boolean, optional
            Default = True, activates tqdm progress timer
            
        Attributes
        ----------
        tracks : pd.DataFrame
            Dataframe with columns x,y,t,name,track
            Contains trajectories based on points in `starts`
        '''
        
        # Check if track dataframe needs to be created
        if hasattr(self,'tracks') == False:
            self.tracks = pd.DataFrame()
        
        # Set up iterator
        if timer:
            iterator = tqdm.tqdm(starts.index)
        else:
            iterator = starts.index
            
        for i in iterator:
            x0,y0 = starts.iloc[i]
            track = self.calc_track(x0,y0,dt,tmin)
            trackdf = pd.DataFrame({'x':track[0,:],'y':track[1,:],'t':self.tval,
                                    'track':[i]*track.shape[-1],
                                    'name':[name]*track.shape[-1]})
            
            self.tracks = pd.concat([self.tracks,trackdf])
            

def load_avi_as_array(path):
    '''
    Use `av` module to load each frame from an avi movie
    into a numpy array
    
    Parameters
    ----------
    path : str
        Complete or relative path to avi movie file for import
        
    Returns
    -------
    np.array 
        Array with dimensions frames,x,y
    '''
    
    # Import movie data
    v = av.open(path)
    
    # Initialize list to save each frame
    Larr = []
    
    # Save each frame as an array
    for packet in v.demux():
        for frame in packet.decode():
            Larr.append(np.asarray(frame.to_image()))
            
    # Convert list of arrays to single array
    vimg = np.array(Larr)
    
    return(vimg)
            
def make_track_movie(movie,df,c,name):
    '''
    Plots the trajectory of points over time on each frame of 
    an existing movie or array
    
    Parameters
    ----------
    movie : str
        Complete or relative path to the movie file to plot on
    df : pd.DataFrame
        Dataframe of tracks minimally with columns x,y,t
    c : str,color
        Currently only supports single color assignments, 
        but data specific assignments could be possible
    name : str
        Root of filename for output file, without filetype 
        
    Returns
    -------
    Saves a tif stack using path provided by `name`
    '''
    
    # Import specialized plotting functions for non-gui backend
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    
    # If input is a string, import movie from specified path
    if type(movie) == str:
        vimg = load_avi_as_array(movie)
    # If input not a string and not an array, print error message
    elif type(movie) == np.ndarray:
        vimg = movie
    else:
        print('movie input must be a path to avi or a numpy array')
        return()
    
    # Initialize list to save arrays of each frame
    Larr = []
    
    # Generate plot of each frame and save to list
    for t in tqdm.tqdm(range(vimg.shape[0])):
        # Setup figure object and subplot axes
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        
        # Plot image and tracks
        ax.imshow(vimg[t])
        ax.scatter(df[df.t==t].x,df[df.t==t].y,c=c)
        
        # Format figure for output
        fig.tight_layout(pad=0)
        ax.axis('off')
        
        # Draw plot which is required for saving
        canvas.draw()
        
        # Esport plotting figure as a string of rgb values
        rgbstring = canvas.tostring_rgb()
        
        # Load rgb string into a numpy array
        image = np.frombuffer(rgbstring,dtype='uint8')
        
        # Reshape image array to fit dimensions of the original plot
        Larr.append(
            image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        )
        
    # Compile array list to array and save
    tifffile.imsave(name+'.tif',data=np.array(Larr))