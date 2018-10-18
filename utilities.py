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