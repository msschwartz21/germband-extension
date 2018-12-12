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
.. warning:: Write about parameter sweep
    
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

.. warning:: write about data wrangling for vector data

``OpticalFlow.m`` returns 5 data objects that are saved to output files when using ``OpticalFlowOutput.m``. 

Simulated Cell Tracking
------------------------

Managing time units
^^^^^^^^^^^^^^^^^^^^^
When we run the optical flow algorithm, it takes a parameter for :math:`\delta t` however, an initial exploration of the role of this parameter indicates that changing the value does not impact the output of the algorithm. This finding raises the question of what are the units for the vectors output by the algorithm. We will use dimensional analysis of the advection equation to determine the units.

The advection equation describes the movement of a particle carried by a flow 

.. math::
    \Delta I = I(t+\Delta t) - I(t) = -\Delta t(\mathbf{v}\cdot\nabla I)

:math:`\Delta I` and :math:`\nabla I` are defined in arbitrary units (:math:`\text{au}`) and describe the change in intensity and the gradient of intensity respectively. The unit of time is not explicitly specified, but we will measure it in seconds (:math:`\text{s}`). Given these units, we can calculate the units of :math:`\mathbf{v}` (:math:`[\mathbf{v}]`).

.. math:: 

    \begin{align}
        \text{au} &= \text{s}([\textbf{v}]\cdot\text{au}) \\
        \frac{\text{au}}{\text{au}} &= \text{s}[\mathbf{v}] \\
        \frac{1}{\text{s}} &= [\mathbf{v}] \\
    \end{align}
    
We find that our velociy vector :math:`\mathbf{v}` has units of :math:`1/\text{s}`. When we are calculating the movement of a simulated cell through our fector field, we need to multiply the velocity by the time step in order to calculate the change in position.

.. math::

    \begin{align}
        \mathbf{x}_n = 
        \begin{bmatrix}
            x_n \\
            y_n \\
        \end{bmatrix} \\
        \mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t \times \mathbf{v}
    \end{align}
    
Currently, the function that interpolates the vector fields for simulated cell tracking :class:`gbeflow.VectorField` relies on ``scipy``'s `interpolate.RectBivariateSpline <rbv_>`_ to estimate the vector for each simulated cell. When :func:`gbeflow.VectorField.initialize_interpolation` runs, it saves the ``scipy`` interpolation object to :attr:`gbeflow.VectorField.Ldx` or :attr:`gbeflow.VectorField.Ldy` for each timepoint. In order to calculate the trajectory of a single cell, the function :func:`gbeflow.VectorField.calc_track` uses the previously calculated interpolation objects stored in :attr:`gbeflow.VectorField.Ldx` and :attr:`gbeflow.VectorField.Ldy` and evaluates the position of the cell to generate the :math:`x` and :math:`y` components of the next vector.

.. _rbv: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html

.. _vig: https://www.sciencedirect.com/science/article/pii/S0006349516300339?via%3Dihub