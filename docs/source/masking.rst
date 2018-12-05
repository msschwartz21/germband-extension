Embryo Masking
================

Rationale
----------
We want to eliminate the background by segmenting the embryo from the background. Later in our analysis, our job will be easier if we do not need to worry about the background contributing noise.

Basic Approach
---------------
See final_contour_based_segmentation.ipynb_ for details on the contour based approach. 

Limitations
-------------
While this approach worked well on the test dataset, when applied to other embryos it would make mistakes on ~10% of frames. There does not appear to be a straightforward way to improve performance without manual curation. See endpoint_ellipse.ipynb_ and vector_calculation.ipynb_ for examples of masking mistakes.

.. _final_contour_based_segmentation.ipynb: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181017-final_contour_based_segmentation.ipynb

.. _endpoint_ellipse.ipynb: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181106-endpoint_ellipse.ipynb

.. _vector_calculation.ipynb: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181108-vector_calculation.ipynb

Next Steps
-----------
The implementation of contour masking shown in vector_calculation.ipynb_ fits an individual contour to create a mask for each timepoint in the timecourse. Simply by repeating the calculation many times, we introduce more opportunities for mistakes. In the future the contour should be fit just to the first timepoint to create a mask that the optical flow algorithm can use on all timepoints (See the paper's supplement_ for more information.) To accommodate any minute changes in embryo size, we could try to increase the size of the mask by 10%, but given the large scale of the movements that we are interested in it may not be necessary.

.. _supplement: https://ars.els-cdn.com/content/image/1-s2.0-S0006349516300339-mmc9.pdf