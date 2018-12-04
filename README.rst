README: Germband Extension
===========================

Efforts to track the rate of extension of the Drosophila germ band

Usage
-------

API
-----
Documentation is available on `Read the Docs`__.

.. _rtd: https://germband-extension.readthedocs.io/en/latest/
__ rtd_

Prerequisites
--------------

Anaconda
^^^^^^^^^
`Anaconda <https://www.anaconda.com/>`_ is a distribution and package manager targeted for scientific computing. While Anaconda mostly focuses on python, it is capable with interfacing with other languages/packages as well. Several of the packages required for gbeflow have underlying C dependences that make independent installation difficult. Anaconda includes all of these packages and makes it simple to install any additional packages. gbeflow is written in Python 3, so we recommend installing the most up-to-date Anaconda installation for Python 3.

Matlab
^^^^^^^
The optical flow algorithm which gbeflow relies on is written in Matlab `(Vig et al. 2016)`__. For more information on optical flow and the algorithm, checkout :ref:`opticalflow`. For the purposes of installation and setup, all you need to know is that you need a local installation of Matlab on your computer to run steps involving the optical flow algorithm. gbeflow was developed using Matlab 2017b, but there are not any dependences known to this specific version. We will run Matlab scripts out of python so no Matlab knowledge is required. However, Matlab is currently only compatible with Python 3.6, so we will need to set up an environment to specifically run 3.6.

.. _vig: https://www.sciencedirect.com/science/article/pii/S0006349516300339?via%3Dihub

__ vig_

Environment Setup
^^^^^^^^^^^^^^^^^^
For simplicity, there will (hopefully) be an environment file, which enables you to setup a virtual environment in Anaconda with all the appropriate dependencies. This file should be downloaded during the installation steps below, but it is also available here_. Checkout the anaconda navigator docs_ for information on how to setup a new environment based on a specification file.

.. _here: https://github.com/msschwartz21/germband-extension/blob/master/environment.yml

.. _docs: https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment

Installation
--------------
The code for gbeflow is hosted on Github_. Users can clone the repository by running the following code in the command line

.. code-block:: bash

    git clone https://github.com/msschwartz21/germband-extension.git
    
Alternatively, the current version of the repository can be downloaded as a zip file `here`__.
    
After cloning the repository, gbeflow can be installed in developer mode by running the following command from the root of the gbeflow directory

.. code-block:: bash

    pip install -e .

.. _Github: https://github.com/msschwartz21/germband-extension

.. _zip: https://github.com/msschwartz21/germband-extension/archive/master.zip

__ zip_


License
--------
gbeflow is licensed under the MIT License.

