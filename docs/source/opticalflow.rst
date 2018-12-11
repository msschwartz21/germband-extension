.. _opticalflow: 

Optical Flow
==============


The Algorithm
--------------

Comparison to PIV
------------------

Implementation in Matlab and Python
------------------------------------
The optical flow algorithm is implemented in a Matlab script published in the supplement to `Vig et al. 2016 <vig_>`_. It can be run directly in Matlab according to the instructions in the supplementary PDF. In order for Matlab to use the script `OpticalFlow.m`, it needs to either be placed in the same folder as the data or the path to the script needs to be added to the Matlab environment.

.. code-block:: matlab
    
    addpath(path/to/script.m)
    
The ``OpticalFlow`` function takes a set of parameters that are described in the supplementary materials of `Vig et al. 2016 <vig_>`_. The parameters are summarized below:
        
.. csv-table:: Optical Flow Parameters (`Vig et al. 2016 <vig_>`_)
    :header: Parameter, Description, Value
    :widths: 15, 70, 15
    
    MovieName, Avi or tiff file to be analyzed, 
    BinaryMask, Name of the region-of-interest image sequence files can be either  AVI or TIFF format. If an ROI mask is not necessary then the input is [ ], [ ]
    scale, Converts pixels to microns scale is defined in microns per pixel, 0.5 µm/pixel
    dt, Time interval between frames, 60 s
    BoxSize, Sets the linear size of the subregions (in pixels) where the velocity is computed. Should be set to be large enough that each subregion contains at least one identifiable image feature., 30 pixels
    BlurSTD, Sets the size of the standard deviation for the Gaussian blur. Should be set to half maximum velocity between two images in pixels., 1
    ArrowSize, Used to define a coarser output grid for the velocity vectors. Defines the spacing (in pixels) between output velocity vectors., 5
    Optical Flow Method, To access an augmented mode of Optical Flow enter ‘Rotation’ to determine the local voriticity (ωο) or ‘React’ to measure the effects of an added source term (γ). To run the standard default version use ‘none’ or omit this input parameter., 
    
The following line of code runs the `OpticalFlow.m` script from the Matlab console.

.. code-block:: matlab

    [X,Y,Vx,Vy,Mov] = OpticalFlow (MovieName, BinaryMask, scale, dt, BoxSize, BlurSTD, ArrowSize)

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