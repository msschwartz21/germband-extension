import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_hyperstack(fpath,dataset='channel1',tmax=166): 
    
    f = h5py.File(fpath,'r')
    L = []
    for t in range(tmax+1):
        L.append(np.array(f.get('t'+str(t)).get(dataset)))
        
    return(np.stack(L))

def imshow(img,figsize=(10,8)):
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