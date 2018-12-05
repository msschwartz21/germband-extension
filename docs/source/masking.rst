Embryo Masking
================

Rationale
----------
We want to eliminate the background by segmenting the embryo from the background. Later in our analysis, our job will be easier if we do not need to worry about the background contributing noise.

Basic Approach
---------------

.. include:: 20181017-final_contour_based_segmentation.rst

Following the development of the method above, an object was created within gbeflow to handle the tasks associated with contouring: :class:`MaskEmbryo`. An example of using this new class is included below.

.. code-block:: python

    import gbeflow
    import tifffile
    import bebi103
    
    # Import sample from tiff
    img = tifffile.imread(filepath)
    
    # Select endpoints of embryo for ellipse calculation
    clk = bebi103.viz.record_clicks(raw[0],flip=False)
    
    # Save points to dataframe after selection
    points = clk.to_df()
    
    # Initialize object with selected points
    me = gbeflow.MaskEmbryo(points)
    
    # Contour embryo based on ellipse calculated during initialization
    # Input accepts only 2d data
    mask = me.contour_embryo(img[0])
    
    # Apply binary mask to the raw image
    mask_img = me.mask_image(img,mask)
    
After running the code above, this workflow can be applied to the remaining timepoints and saved as a tif.

Limitations
-------------
While this approach worked well on the test dataset, when applied to other embryos it would make mistakes on ~10% of frames. There does not appear to be a straightforward way to improve performance without manual curation. See `20181106-endpoint_ellipse.ipynb <endpoint_>`_ and `20181108-vector_calculation.ipynb <vectorcalc_>`_ for examples of masking mistakes.

.. _contourntbk: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181017-final_contour_based_segmentation.ipynb

.. _endpoint: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181106-endpoint_ellipse.ipynb

.. _vectorcalc: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181108-vector_calculation.ipynb

Next Steps
-----------
The implementation of contour masking shown in `20181108-vector_calculation.ipynb <vectorcalc_>`_ fits an individual contour to create a mask for each timepoint in the timecourse. Simply by repeating the calculation many times, we introduce more opportunities for mistakes. In the future the contour should be fit just to the first timepoint to create a mask that the optical flow algorithm can use on all timepoints (See the paper's supplement_ for more information.) To accommodate any minute changes in embryo size, we could try to increase the size of the mask by 10%, but given the large scale of the movements that we are interested in it may not be necessary.

.. _supplement: https://ars.els-cdn.com/content/image/1-s2.0-S0006349516300339-mmc9.pdf