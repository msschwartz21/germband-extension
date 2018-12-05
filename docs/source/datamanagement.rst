Data Management
================

Import raw microscopy data
----------------------------
Data can be imported from czi and tiff files. The :class:`CziImport` wraps the czifile_ module published by Christoph Gohlke to read the propriatory zeiss file format. The underlying datastructure of czi's varies based on the collection parameters, so the :class:`CziImport` may not work reliabily. Alternativly, data can be read from tiff files saved using Fiji_. The module tifffile_ makes this by far the easiest approach with ``tifffile.imread`` and ``tifffile.imsave``.

Embryo alignment
-----------------
During imaging, embryos are not expected to be aligned in any particular orientation in the XY plane. While we will accept only embryos that have an approximate lateral mounting, we need to correct XY positioning in post-processing. In the notebook rotate_embryo_, a workflow is proposed that accepts user inputs in order to guide the alignment process. The user input functions rely on  ``bebi103`` which is a package written by Justin Bois for the BE/Bi103 course that is still under active development.

The final result of this workflow is that all samples are aligned dorsal up with the anterior end of the embryo to the left. This consistent alignment should facilitate future comparisons of the results of optical flow. 

Optical flow outputs
---------------------

Optical flow transformations
------------------------------

.. _rotate_embryo: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181116-rotate_embryo.ipynb

.. _gbe_mutant_processing: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181203-gbe_mutant_processing.ipynb

.. _czifile: https://github.com/AllenCellModeling/czifile

.. _Fiji: https://fiji.sc/

.. _tifffile: https://pypi.org/project/tifffile/