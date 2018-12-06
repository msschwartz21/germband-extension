Approach
----------

.. code:: ipython3

    import os
    import glob
    import tqdm
    
    import numpy as np
    import pandas as pd
    import scipy.ndimage
    
    import tifffile
    import bebi103 
    
    import bokeh.io
    bokeh.io.output_notebook()
    
    import gbeflow
    
.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="1001">Loading BokehJS ...</span>
        </div>    

Import first time point from each original tiff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    files = glob.glob(os.path.abspath('../data/original/20181130-gbe_mutants_brt/*.tif'))
    files = files

The ``key`` parameter for ``tifffile.imread`` lets us import just the
first timepoint with both channels. This makes it possible so that we
can look at a sample of all of the data at once. Since the total dataset
is >20Gb we can't load it directly into memory.

.. code:: ipython3

    %%time
    raw = {}
    for f in files:
        raw[f] = tifffile.imread(f,key=(0,1))


.. parsed-literal::

    CPU times: user 524 ms, sys: 102 ms, total: 626 ms
    Wall time: 177 ms


Select points to use for alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Try selecting two points on the dorsal surface that represent the linear
plane of the dorsal side. This approach should hopefully be less
sensitive to variability in how the user picks the points.

.. code:: ipython3

    clicks = []
    for f in files:
        clk = bebi103.viz.record_clicks(raw[f][1],flip=False)
        clicks.append(clk)



.. raw:: html

    
    <script src="http://localhost:58684/autoload.js?bokeh-autoload-element=1003&bokeh-absolute-url=http://localhost:58684&resources=none" id="1003"></script>



.. raw:: html

    
    <script src="http://localhost:58685/autoload.js?bokeh-autoload-element=1005&bokeh-absolute-url=http://localhost:58685&resources=none" id="1005"></script>



.. raw:: html

    
    <script src="http://localhost:58686/autoload.js?bokeh-autoload-element=1007&bokeh-absolute-url=http://localhost:58686&resources=none" id="1007"></script>



.. raw:: html

    
    <script src="http://localhost:58688/autoload.js?bokeh-autoload-element=1009&bokeh-absolute-url=http://localhost:58688&resources=none" id="1009"></script>

Extract the points selected for each image into a dataframe.

.. code:: ipython3

    Ldf = []
    for clk in clicks:
        Ldf.append(clk.to_df())
        
    points = pd.concat(Ldf,keys=files)
    points




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>x</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_05_eve.tif</th>
          <th>0</th>
          <td>583.782051</td>
          <td>811.211429</td>
        </tr>
        <tr>
          <th>1</th>
          <td>819.794872</td>
          <td>311.188571</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_21_kr.tif</th>
          <th>0</th>
          <td>182.525641</td>
          <td>412.960000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>380.769231</td>
          <td>829.177143</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_17_kr.tif</th>
          <th>0</th>
          <td>368.294872</td>
          <td>650.411429</td>
        </tr>
        <tr>
          <th>1</th>
          <td>550.384615</td>
          <td>206.422857</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_23_kr.tif</th>
          <th>0</th>
          <td>334.474359</td>
          <td>685.405714</td>
        </tr>
        <tr>
          <th>1</th>
          <td>352.769231</td>
          <td>197.600000</td>
        </tr>
      </tbody>
    </table>
    </div>


Reshape ``points`` array to have one row per sample.

.. code:: ipython3

    points = points.reset_index(level=1)
    points.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>level_1</th>
          <th>x</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_05_eve.tif</th>
          <td>0</td>
          <td>583.782051</td>
          <td>811.211429</td>
        </tr>
        <tr>
          <th>/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_05_eve.tif</th>
          <td>1</td>
          <td>819.794872</td>
          <td>311.188571</td>
        </tr>
        <tr>
          <th>/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_21_kr.tif</th>
          <td>0</td>
          <td>182.525641</td>
          <td>412.960000</td>
        </tr>
        <tr>
          <th>/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_21_kr.tif</th>
          <td>1</td>
          <td>380.769231</td>
          <td>829.177143</td>
        </tr>
        <tr>
          <th>/Users/morganschwartz/Code/germband-extension/data/original/20181130-gbe_mutants_brt/20181130-gbe_mutants_brt_17_kr.tif</th>
          <td>0</td>
          <td>368.294872</td>
          <td>650.411429</td>
        </tr>
      </tbody>
    </table>
    </div>



Calculate a line for each embryo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::  y - y_1 = m(x - x_1) 

.. math::  m = \frac{y_2 - y_1}{x_2 - x_1} 

.. code:: ipython3

    line = gbeflow.calc_line(points)

.. code:: ipython3

    line = line.reset_index().rename(columns={'index':'f'})
    line.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>f</th>
          <th>x1</th>
          <th>x2</th>
          <th>y1</th>
          <th>y2</th>
          <th>dx</th>
          <th>dy</th>
          <th>m</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>583.782051</td>
          <td>819.794872</td>
          <td>811.211429</td>
          <td>311.188571</td>
          <td>236.012821</td>
          <td>-500.022857</td>
          <td>-2.118626</td>
        </tr>
        <tr>
          <th>1</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>182.525641</td>
          <td>380.769231</td>
          <td>412.960000</td>
          <td>829.177143</td>
          <td>198.243590</td>
          <td>416.217143</td>
          <td>2.099524</td>
        </tr>
        <tr>
          <th>2</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>368.294872</td>
          <td>550.384615</td>
          <td>650.411429</td>
          <td>206.422857</td>
          <td>182.089744</td>
          <td>-443.988571</td>
          <td>-2.438295</td>
        </tr>
        <tr>
          <th>3</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>334.474359</td>
          <td>352.769231</td>
          <td>685.405714</td>
          <td>197.600000</td>
          <td>18.294872</td>
          <td>-487.805714</td>
          <td>-26.663522</td>
        </tr>
      </tbody>
    </table>
    </div>



Plot embryos with line on top
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::  y = m(x - x_1) + y_1

.. code:: ipython3

    # Create list to collect plot objects
    Lp = []
    
    # X values to compute line on
    x = np.linspace(0,1024,100)
    
    for f in files:
        p = bebi103.viz.imshow(raw[f][1,:,:],flip=False)
        
        x1 = line[line['f']==f]['x1'].values
        y1 = line[line['f']==f]['y1'].values
        m = line[line['f']==f]['m'].values
        y = m*(x-x1) + y1
        
        p.line(x,y,color='red',line_width=3)
    #     p.scatter(line[line['f']==f]['x'], line[line['f']==f]['y'],color='white',size=15)
        
        Lp.append(p)
        
    bokeh.io.show(bokeh.layouts.gridplot(Lp,ncols=2))



.. raw:: html

    
    
    
    
    
    
      <div class="bk-root" id="d141fbb7-255e-4150-8635-f74c54d7a613"></div>







Calculate rotation
^^^^^^^^^^^^^^^^^^^^^^

The angle of rotation is calculated as follows

.. math::  \theta = \arctan\bigg(\frac{y_2-y_1}{x_2-x_1}\bigg)

This calculation can be coded using the ``np.arctan2``, which has two arguments that correspond to :math:`\Delta y` and :math:`\Delta x`.

.. code:: ipython3

    line = gbeflow.calc_embryo_theta(line)
    line.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>f</th>
          <th>x1</th>
          <th>x2</th>
          <th>y1</th>
          <th>y2</th>
          <th>dx</th>
          <th>dy</th>
          <th>m</th>
          <th>theta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>583.782051</td>
          <td>819.794872</td>
          <td>811.211429</td>
          <td>311.188571</td>
          <td>236.012821</td>
          <td>-500.022857</td>
          <td>-2.118626</td>
          <td>-64.732499</td>
        </tr>
        <tr>
          <th>1</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>182.525641</td>
          <td>380.769231</td>
          <td>412.960000</td>
          <td>829.177143</td>
          <td>198.243590</td>
          <td>416.217143</td>
          <td>2.099524</td>
          <td>64.531611</td>
        </tr>
        <tr>
          <th>2</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>368.294872</td>
          <td>550.384615</td>
          <td>650.411429</td>
          <td>206.422857</td>
          <td>182.089744</td>
          <td>-443.988571</td>
          <td>-2.438295</td>
          <td>-67.700358</td>
        </tr>
        <tr>
          <th>3</th>
          <td>/Users/morganschwartz/Code/germband-extension/...</td>
          <td>334.474359</td>
          <td>352.769231</td>
          <td>685.405714</td>
          <td>197.600000</td>
          <td>18.294872</td>
          <td>-487.805714</td>
          <td>-26.663522</td>
          <td>-87.852162</td>
        </tr>
      </tbody>
    </table>
    </div>



Apply rotation based on :math:`\theta`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`\theta` calculated, we are now ready to rotate each sample
accordingly. Since we cannot load all of the data into memory at the
same time, we will currently only rotate the first timepoint to check
that it worked. After we have determined all necessary manipulations for
each embryo, we will run the actual rotation.

.. code:: ipython3

    # Dataframe to save first timepoint from each rotate embryo
    rot = {}
    
    # List to save bokeh plots
    Lp = []
    
    for f in tqdm.tqdm(files):
        # Extract the theta value for this sample
        theta = line[line['f']==f]['theta'].values[0]
        
        # Rotate single image
        rimg = scipy.ndimage.rotate(raw[f][1],theta)
        
        # Save and plot first timepoint
        rot[f] = rimg
        p = bebi103.viz.imshow(rimg,title=f)
        Lp.append(p)
        


.. parsed-literal::

    100%|██████████| 4/4 [00:00<00:00,  6.63it/s]


.. code:: ipython3

    bokeh.io.show(bokeh.layouts.gridplot(Lp,ncols=2))



.. raw:: html

    
    
    
    
    
    
      <div class="bk-root" id="0b076df3-ecf7-4868-9775-abfae3ec06b8"></div>




