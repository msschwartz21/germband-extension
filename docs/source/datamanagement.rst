Data Preprocessing
====================

Import raw microscopy data
----------------------------
Data can be imported from czi and tiff files. The :class:`gbeflow.CziImport` wraps the czifile_ module published by Christoph Gohlke to read the propriatory zeiss file format. The underlying datastructure of czi's varies based on the collection parameters, so the :class:`gbeflow.CziImport` may not work reliabily. 

.. code-block:: python

    import gbeflow
    
    # Load czi file from relative/absolute path
    czi = gbeflow.CziImport(filepath)
    
    # Raw array of data is available as an attribute
    raw = czi.raw_im
    
    # Data squeezed to minimum dimensions also available
    data = czi.data

Alternativly, data can be read from tiff files saved using Fiji_. The module tifffile_ makes this by far the easiest approach with ``tifffile.imread`` and ``tifffile.imsave``.

.. code-block:: python

    import tifffile
    
    # Import tiff file from relative or absolute path
    img = tifffile.imread(filepath)
    
    # Save array to new tiff file
    tifffile.imsave(outpath, data=img)

Embryo alignment
-----------------
During imaging, embryos are not expected to be aligned in any particular orientation in the XY plane. While we will accept only embryos that have an approximate lateral mounting, we need to correct XY positioning in post-processing. In the notebook rotate_embryo_, a workflow is proposed that accepts user inputs in order to guide the alignment process. The user input functions rely on bebi103_ which is a package written by Justin Bois for the BE/Bi103 course that is still under active development. Below is an example of what processing a single sample might look like.

.. include:: 20181203-gbe_mutant_processing.rst

.. code-block:: python

    # Import modules
    import bebi103
    import gbeflow
    import tifffile
    import scipy.ndimage
    
    # Import image from tiff
    raw = tifffile.imread(impath)

    # Select first timepoint from `raw` to display
    clk = bebi103.viz.record_clicks(raw[0],flip=False)
    
    # Extract points from `clk` to a dataframe
    points = clk.to_df().reset_index()
    
    # Calculate line for each embryo
    line = gbeflow.calc_line(points
                    ).reset_index(
                    ).rename(
                        columns={'index','f'}
                    )
                    
    # Calculate angle of rotation
    line = gbeflow.calc_embryo_theta(line)
    
    # Rotate embryo 
    rimg = scipy.ndimage.rotate(raw,line['theta'])
    
    # Save rotated stack to file
    tifffile.imsave(outpath,data=rimg)
    
Manual curation of orientation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the rotation based on :math:`\theta` has been applied, the embryos should be positioned such that the AP axis is horizontal. However, the rotation does not gurantee that the embryo will be positioned with anterior left and dorsal up. At this point, the user can individually specify any additional rotations to correct the embryo's orientation. The following examples are typically sufficient to correct most orientation errors:

.. code-block:: python

    # Rotate by 180 degrees around center point
    img = scipy.ndimage.rotate(img,180)
    
    # Flip horizontally by specifying a specific axis
    img = np.flip(img,axis=-1)
    
    # Flip vertically
    img = np.flip(img,axis=-1)

The final result of this workflow is that all samples are aligned dorsal up with the anterior end of the embryo to the left. This consistent alignment should facilitate future comparisons of the results of optical flow. 

.. include:: masking.rst



.. _rotate_embryo: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181116-rotate_embryo.ipynb

.. _gbe_mutant_processing: https://github.com/msschwartz21/germband-extension/blob/master/notebooks/20181203-gbe_mutant_processing.ipynb

.. _czifile: https://github.com/AllenCellModeling/czifile

.. _Fiji: https://fiji.sc/

.. _tifffile: https://pypi.org/project/tifffile/

.. _bebi103: https://github.com/justinbois/bebi103