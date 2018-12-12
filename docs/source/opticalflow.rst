.. _opticalflow: 

Optical Flow
==============

Implementation in Matlab
------------------------------------
The optical flow algorithm is implemented in a Matlab script published in the supplement to `Vig et al. 2016 <vig_>`_. It can be run directly in Matlab according to the instructions in the supplementary PDF. In order for Matlab to use the script `OpticalFlow.m`, it needs to either be placed in the same folder as the data or the path to the script needs to be added to the Matlab environment before running the script.

.. code-block:: matlab
    
    addpath('path/to/OpticalFlow.m')
    [X,Y,Vx,Vy,Mov] = OpticalFlow (MovieName, BinaryMask, scale, dt, BoxSize, BlurSTD, ArrowSize)
    
The ``OpticalFlow`` function takes a set of parameters that are described in the supplementary materials of `Vig et al. 2016 <vig_>`_. The parameters are summarized below:
        
.. csv-table:: Optical Flow Parameters (`Vig et al. 2016 <vig_>`_)
    :header: Parameter, Description, Value
    :widths: 15, 70, 15
    
    MovieName, Avi or tiff file to be analyzed, 
    BinaryMask, Name of the region-of-interest image sequence files can be either  AVI or TIFF format. If an ROI mask is not necessary then the input is [ ], [ ]
    scale, Converts pixels to microns scale is defined in microns per pixel, 0.5 µm/pixel (must be a float)
    dt, Time interval between frames, 60.0 s (must be a float)
    BoxSize, Sets the linear size of the subregions (in pixels) where the velocity is computed. Should be set to be large enough that each subregion contains at least one identifiable image feature., 30 pixels
    BlurSTD, Sets the size of the standard deviation for the Gaussian blur. Should be set to half maximum velocity between two images in pixels., 1.0 (must be a float)
    ArrowSize, Used to define a coarser output grid for the velocity vectors. Defines the spacing (in pixels) between output velocity vectors., 5
    Optical Flow Method, To access an augmented mode of Optical Flow enter ‘Rotation’ to determine the local voriticity (ωο) or ‘React’ to measure the effects of an added source term (γ). To run the standard default version use ‘none’ or omit this input parameter., 
    
Selecting Parameter Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^

BlurSTD and BoxSize are the two parameters that are most difficult to determine for a naive user. In order to determine appropriate values for brightfield data, I tried running combinations of BoxSize={10,20,30,40,50} and BlurSize={1,4,10,20,30} and reviewed the outputs to select parameter values that had the least noise without obscuring features in the data. This same approach could be applied to any new data types that are investigated in the future.
    
Running OpticalFlow from python
--------------------------------
Matlab distributes its application with a python installation, which can be installed by the user. See `this link <matlabengine_>`_ for instructions on installing the matlab python package.

Managing the output of OpticalFlow.m
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``OpticalFlow.m`` returns a set of variables as output: ``[X,Y,Vx,Vy,Mov]``. Transfering matlab data objects directly to python is difficult and limited to a relatively simple data structures. To avoid this problem, I wrote a function that wraps around ``OpticalFlow.m`` and saves the output to a set of csv files and an avi. The code shown below is available in `OpticalFlowOutput.m`_.

.. literalinclude:: ../../matlab/OpticalFlowOutput.m
   :language: matlab

.. _OpticalFlowOutput.m: https://github.com/msschwartz21/germband-extension/blob/master/matlab/OpticalFlowOutput.m

Interacting with matlab from python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the matlab package is installed in the ``python36`` environment, we can run ``OpticalFlow.m`` script directly from python using a Jupyter Notebook. 

.. code-block:: python

    # First we need to import the matlab engine package
    import matlab.engine
    
    # Next start the engine
    eng = matlab.engine.start_matlab()
    
    # Add path for matlab script to namespace
    eng.addpath(r'../matlab',nargout=0)
    
    # Define opticalflow parameter values
    BinaryMask = matlab.single([])
    scale = 0.5
    dt = 1.0
    BoxSize = 30
    BlurSTD = 1.0
    ArrowSize= 5
    
    # Define file paths and names
    outpath = 'abs/path/name'
    inpath = 'abs/path/name.tif'
    
    # Run optical flow script
    eng.OpticalFlowOutput(outpath, inpath, BinaryMask,
                        BoxSize, BlurSTD, ArrowSize,
                        scale, dt, nargout=0)
                        
.. _matlabengine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
                        
Some notes about running matlab from python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When Matlab functions are called from Python, we need to include an additional arguement ``nargout``, which tells Matlab whether we expect a value returned by the function. For the functions ``eng.addpath`` and ``eng.OpticalFlowOutput``, there are no return values so we can just append ``nargout=0`` to the function parameters as shown above.

There are two options for the parameter BinaryMask. If we want to use a previously calculated binary mask, we can pass the absolute path to an avi or tiff file that matches the primary file that we are analyzing. Alternatively, if we don't want to use a mask, we can set BinaryMask to ``matlab.single([])``, which passes an empty array to Matlab.

Finally, the ``OpticalFlow.m`` script and Matlab seem happiest when working with absolute paths as opposed to relative paths. For convenience, we can import the package ``os`` and use the function ``os.path.abspath()`` to convert any relative or complete path to an absolute path. For example:

.. code-block:: python

    import os
    
    relpath = '../data/example.tif'
    abspath = os.path.abspath(relpath)

Wrangling the optical flow ouput data
--------------------------------------

``OpticalFlow.m`` returns 5 data objects ``[X,Y,Vx,Vy,mov]`` that are saved to output files when using ``OpticalFlowOutput.m``. Given a user input which defines the base of the output file names hereafter refered to as ``<name>``, ``OpticalFlowOutput`` saves the 5 data objects.

.. csv-table:: Optical Flow Output
    :header: Object, Filename, Description
    :widths: 25, 25, 50
    
    X, <name>_X.csv, A vector with the unique x positions for the vector grid 
    Y, <name>_Y.csv, A vector with the unique y positions for the vector grid
    Vx, <name>_Vx.csv, Contains the X component of the velocity vector. An MxN matrix where there are M columns corresponding to time and N rows corresponding the the spatial positions stored in X and Y
    Vy, <name>_Vy.csv, Contains the Y component of the velocity vector. An MxN matrix where there are M columns corresponding to time and N rows corresponding the the spatial positions stored in X and Y
    mov, <name>.avi, A movie of the original input data with the corresponding vector field overlaid using green arrows.
    
The function :func:`gbeflow.tidy_vector_data()` loads the 4 vector files saved by optical flow ouput by looking for the root of the filename <name>. The four data files are compiled into a single dataframe that contains five columns: frame, x, y, vx, and vy. The dataframe which the function returns can be saved to a csv file using ``pandas.DataFrame.to_csv()``.

In order to facilitate interpolation and plotting, we also want to transform the data into an array based structure as opposed to a tidy dataframe where each row corresponds to a single point/velocity vector. The function :func:`gbeflow.reshape_vector_data` accepts the dataframe output by :func:`gbeflow.tidy_vector_data` as an input. It creates a set of arrays that conform to the following dimensions: # of time points :math:`\times` # of unique x values :math:`\times` # of unique y values. The function returns five arrays following this convention: ``tt``, ``xx``, ``yy``, ``vx``, and ``vy``. If we use the same index to select a value from each of the 5 arrays, we will get the t, x and y positions with the corresponding vx and vy velocity components.

The following code is an example of how to import the results of optical flow after running the example code above.

.. code-block:: python

    import gbeflow

    # Define file paths and names
    outpath = 'abs/path/name'
    
    # Read in data to a single dataframe
    df = gbeflow.tidy_vector_data(outpath)
    
    # Convert to array based format
    tt,xx,yy,vx,vy = gbeflow.reshape_vector_data(df)