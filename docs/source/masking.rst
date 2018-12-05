Embryo Masking
================

Rationale
----------

Basic Approach
---------------
See `20181017-final_contour_based_segmentation`_ for details on the contour based approach. 

Limitations
-------------
While this approach worked well on the test dataset, when applied to other embryos it would make mistakes on ~10% of frames. There does not appear to be a straightforward way to improve performance without manual curation. See `20181106-endpoint_ellipse`_ and `20181108-vector_calculation`_ for examples of masking mistakes.