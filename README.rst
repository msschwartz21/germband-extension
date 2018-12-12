README: Germband Extension
===========================

Efforts to track the rate of extension of the Drosophila germ band

Usage
-------

Prerequisites
--------------

Anaconda
^^^^^^^^^
`Anaconda <https://www.anaconda.com>`_ is a distribution and package manager targeted for scientific computing. While Anaconda mostly focuses on python, it is capable with interfacing with other languages/packages as well. Several of the packages required for gbeflow have underlying C dependences that make independent installation difficult. Anaconda includes all of these packages and makes it simple to install any additional packages. gbeflow is written in Python 3, so we recommend installing the most up-to-date Anaconda installation for Python 3.


.. _matlabsetup:

Matlab Setup
^^^^^^^^^^^^^
The optical flow algorithm which gbeflow relies on is written in Matlab `(Vig et al. 2016) <vig_>`_. For more information on optical flow and the algorithm, checkout :ref:`opticalflow`. For the purposes of installation and setup, all you need to know is that you need a local installation of Matlab on your computer to run steps involving the optical flow algorithm. gbeflow was developed using Matlab 2017b, but there are not any dependences known to this specific version. We will run Matlab scripts out of python so no Matlab knowledge is required. However, Matlab is currently only compatible with Python 3.6, so we will need to set up an environment to specifically run 3.6. Once we have the correct environment set up, we will install a matlab engine for python.

.. _vig: https://www.sciencedirect.com/science/article/pii/S0006349516300339?via%3Dihub

Setup
--------------
The code for gbeflow is hosted on Github_. Users can clone the repository by running the following code in the command line

.. code-block:: bash

    git clone https://github.com/msschwartz21/germband-extension.git
    
Alternatively, the current version of the repository can be downloaded as a zip file `here <zip_>`_.

.. _Github: https://github.com/msschwartz21/germband-extension

.. _zip: https://github.com/msschwartz21/germband-extension/archive/master.zip

Python Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^
For simplicity, there is a script that enables you to setup a virtual environment in Anaconda with all the appropriate dependencies. This file should be downloaded during the installation steps below, but it is also available `here <setupenv_>`_. To setup the environment run the following commands in your terminal from the root of the germband-extension directory.

.. code-block:: bash

    # Create python3.6 virtual environment
    conda create -n python36 python=3.6 anaconda h5py bokeh tqdm numpydoc
    
    # Activate new environment
    conda activate python36
    
    # Install remaining conda packages 
    conda install nodejs
    conda install -c conda-forge av altair OpenPIV
    
    # Install pip packages
    pip install czifile tifffile sphinx-rtd-theme bebi103 nb_conda_kernels

This script will create an anaconda virtual environment named python36. You can activate the environment by running ``conda activate python36``. When you are done with the environment run ``conda deactivate`` to return to the basic python environment.

.. warning:: ``conda activate`` doesn't appear to work in powershell, but it does work from the Anaconda Prompt and Command.

When we are working in Jupyter Notebooks, nb_conda_kernels_ will provide the option to launch the notebook from any available virtual environment including python36. Justin Bois has a great introduction to Jupyter notebooks available `here <bebi103_>`_.

.. _nb_conda_kernels: https://github.com/Anaconda-Platform/nb_conda_kernels

.. _bebi103: http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2018/tutorials/t0b_intro_to_jupyterlab.html

.. _setupenv: https://github.com/msschwartz21/germband-extension/blob/master/setup_env.sh

.. _docs: https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment

Matlab Engine Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Matlab includes a python engine with its default installation. In order to install the engine as a python module, follow the instructions listed `here <matlabengine_>`_. Make sure that the python 3.6 environment is active by running ``conda activate python36``.

.. _matlabengine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Troubleshooting installation problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These notes are here to serve as a record for previously encountered problems, but undoubtably new problems will show up in the future. When I was working on the installation for the lab computer, Anaconda recommended that PATH variables not be set by Anaconda during the installation process. The result of this choice is that conda/python/etc. can be called from the Anaconda Prompt, but these commands are not available from any other terminal interface such as Command or Powershell. I ended up reinstalling Anaconda and choosing to install with the PATH variables and it hasn't caused any problems to date.

Matlab is typically installed for all users, which means it requires administrator priveleges to make any changes to the directory. As a result, if we use a normal command prompt to try to run ``python setup.py install`` from within the Matlab directory, we are blocked from making changes since normal command interfaces do not have administrator priveleges. To get around this problem, Command or Anaconda Prompt can be launched with administrator priveleges by right clicking on the program to launch and selecting the "Run as Administrator" option. This administrator option should only be used when absolutely necessary, such as running the python installation for matlab.

gbeflow Installation
^^^^^^^^^^^^^^^^^^^^^^
Now that we have a python 3.6 environment setup, we are ready to locally install gbeflow. From the terminal, run the following code to enter the python36 environment and install gbeflow. Begin by navigating to the root of the gbeflow directory and run the following from the command line.

.. code-block:: shell

	$ conda activate python36
	$ pip install -e .
	$ conda deactivate

API
-----
Documentation is available on `Read the Docs <rtd_>`_.

.. _rtd: https://germband-extension.readthedocs.io/en/latest/

License
--------
gbeflow is licensed under the `MIT License <mit_>`_.

.. _mit: https://github.com/msschwartz21/germband-extension/blob/master/LICENSE

