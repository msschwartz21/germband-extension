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

Matlab
^^^^^^^
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

Environment Setup
^^^^^^^^^^^^^^^^^^
For simplicity, there is a script that enables you to setup a virtual environment in Anaconda with all the appropriate dependencies. This file should be downloaded during the installation steps below, but it is also available `here <setupenv_>`_. To setup the environment run the following command in your terminal from the root of the germband-extension directory.

- On windows and macs, ``sh setup_env.sh``

This script will create an anaconda virtual environment named python36. You can activate the environment by running the following:

- On windows, ``activate python36``
- On macs, ``source activate python36``

When you are done with the environment run the following to deactivate:

- On windows, ``deactivate``
- On macs, ``source deactivate``

.. _setupenv: https://github.com/msschwartz21/germband-extension/blob/master/setup_env.sh

.. _docs: https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment

Matlab Installation
^^^^^^^^^^^^^^^^^^^^
Matlab includes a python engine with its default installation. In order to install the engine as a python module, follow the instructions listed `here <matlabengine_>`_. Make sure that the python 3.6 environment is active by running ``source activate python36`` or ``activate python36``.

.. _matlabengine: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Installation
^^^^^^^^^^^^^
Now that we have a python 3.6 environment setup, we are ready to locally install gbeflow. From the terminal, run the following code to enter the python36 environment and install gbeflow. Begin by navigating to the root of the gbeflow directory.

On windows, run the following:

.. code-block:: shell

	$ activate python36
	$ pip install -e .
	$ deactivate

On macs, run the following:

.. code-block:: shell
	
	$ source activate python36
	$ pip install -e .
	$ source deactivate

API
-----
Documentation is available on `Read the Docs <rtd_>`_.

.. _rtd: https://germband-extension.readthedocs.io/en/latest/

License
--------
gbeflow is licensed under the `MIT License <mit_>`_.

.. _mit: https://github.com/msschwartz21/germband-extension/blob/master/LICENSE

