# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: ipynb,py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.0
# ---

import numpy as np
import matplotlib.pyplot as plt
from imp import reload
import sys
sys.path.insert(0, '..')
import utilities as ut

from skimage import img_as_float

hst = ut.read_hyperstack('../data/wt_gbe_20180110.h5')

ut.imshow(hst[0,350:500,100:900])

# # Segmentation to remove background

fhst = img_as_float(hst)

test = fhst[0,350:500,100:900]

ut.imshow(test)

from scipy.ndimage import gaussian_filter

gaus = gaussian_filter(test,4)
ut.imshow(gaus)



from skimage.feature import canny

edges = canny(gaus)
type(edges)

plt.imshow(edges)

from scipy import ndimage as ndi
fill = ndi.binary_fill_holes(edges)
plt.imshow(fill)

from skimage.filters import sobel
elevation_map = sobel(test)
ut.imshow(elevation_map)

markers = np.zeros_like(test)
markers[test<0.1] = 1
markers[test>0.9] = 2
plt.imshow(markers)

from skimage.morphology import watershed

segmentation = watershed(elevation_map,markers)
plt.imshow(segmentation)
