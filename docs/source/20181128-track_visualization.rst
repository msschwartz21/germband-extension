
.. code:: ipython3

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import tifffile
    
    import bebi103
    
    import bokeh.io
    notebook_url = 'localhost:8888'
    bokeh.io.output_notebook()
    
    import os
    import sys
    import glob
    from imp import reload
    import gbeflow


.. parsed-literal::

    /Users/morganschwartz/anaconda3/envs/python36/lib/python3.6/site-packages/bebi103/viz.py:30: UserWarning: DataShader import failed with error "No module named 'datashader'".
    Features requiring DataShader will not work and you will get exceptions.
      Features requiring DataShader will not work and you will get exceptions.""")
    /Users/morganschwartz/anaconda3/envs/python36/lib/python3.6/site-packages/bebi103/viz.py:38: UserWarning: Could not import `stan` submodule. Perhaps pystan is not properly installed.
      warnings.warn('Could not import `stan` submodule. Perhaps pystan is not properly installed.')
    /Users/morganschwartz/anaconda3/envs/python36/lib/python3.6/site-packages/bebi103/__init__.py:19: UserWarning: Could not import `stan` submodule. Perhaps pystan is not properly installed.
      warnings.warn('Could not import `stan` submodule. Perhaps pystan is not properly installed.')



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="1001">Loading BokehJS ...</span>
        </div>




.. code:: ipython3

    csvs = glob.glob('*_Vx.csv')
    csvs




.. parsed-literal::

    ['yolk3_Vx.csv',
     '20180110_htl_glc_sc6_mmzm_rotate_brt_Vx.csv',
     'yolk_Vx.csv',
     'original_Vx.csv',
     'test3_Vx.csv',
     '20180112_htlglc_tl_sc4_resille_rotate_brt_Vx.csv',
     '20180108_htl_glc_sc2_mmzm_wp_rotate_brt_Vx.csv',
     '20180110_htl_glc-CreateImageSubset-02_sc11_htl_rotate_brt_Vx.csv',
     '20180108_htl_glc_sc9_mmzp_rotate_brt_Vx.csv',
     '20180108_htl_glc_sc11_mmzm_rotate_brt_Vx.csv',
     'test2_Vx.csv',
     '20180112_htlglc_tl_sc11_mmzp_rotate_brt_Vx.csv',
     'test_Vx.csv',
     'yolk2_Vx.csv',
     '20180110_htl_glc_sc15_mmzm_rotate_brt_Vx.csv',
     'sc11_Vx.csv',
     '20180110_htl_glc_sc14_mmzp_rotate_brt_Vx.csv',
     'test4_Vx.csv',
     '20180110_htl_glc-CreateImageSubset-01_sc10_wt_rotate_brt_Vx.csv']



.. code:: ipython3

    names = set([f[:-7] for f in csvs])
    names




.. parsed-literal::

    {'20180108_htl_glc_sc11_mmzm_rotate_brt',
     '20180108_htl_glc_sc2_mmzm_wp_rotate_brt',
     '20180108_htl_glc_sc9_mmzp_rotate_brt',
     '20180110_htl_glc-CreateImageSubset-01_sc10_wt_rotate_brt',
     '20180110_htl_glc-CreateImageSubset-02_sc11_htl_rotate_brt',
     '20180110_htl_glc_sc14_mmzp_rotate_brt',
     '20180110_htl_glc_sc15_mmzm_rotate_brt',
     '20180110_htl_glc_sc6_mmzm_rotate_brt',
     '20180112_htlglc_tl_sc11_mmzp_rotate_brt',
     '20180112_htlglc_tl_sc4_resille_rotate_brt',
     'original',
     'sc11',
     'test',
     'test2',
     'test3',
     'test4',
     'yolk',
     'yolk2',
     'yolk3'}



.. code:: ipython3

    fs = ['20180108_htl_glc_sc11_mmzm_rotate_brt',
     '20180108_htl_glc_sc2_mmzm_wp_rotate_brt',
     '20180108_htl_glc_sc9_mmzp_rotate_brt',
     '20180110_htl_glc-CreateImageSubset-01_sc10_wt_rotate_brt',
     '20180110_htl_glc-CreateImageSubset-02_sc11_htl_rotate_brt',
     '20180110_htl_glc_sc14_mmzp_rotate_brt',
     '20180110_htl_glc_sc15_mmzm_rotate_brt',
     '20180110_htl_glc_sc6_mmzm_rotate_brt',
     '20180112_htlglc_tl_sc11_mmzp_rotate_brt',
     '20180112_htlglc_tl_sc4_resille_rotate_brt']

.. code:: ipython3

    # vf = {}
    for f in fs:
        vf[f] = gbeflow.VectorField(f)


.. parsed-literal::

    /Users/morganschwartz/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/indexing.py:1472: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      return self._getitem_tuple(key)


.. code:: ipython3

    vf.keys()




.. parsed-literal::

    dict_keys(['20180108_htl_glc_sc11_mmzm_rotate_brt', '20180108_htl_glc_sc2_mmzm_wp_rotate_brt', '20180108_htl_glc_sc9_mmzp_rotate_brt', '20180110_htl_glc-CreateImageSubset-01_sc10_wt_rotate_brt', '20180110_htl_glc-CreateImageSubset-02_sc11_htl_rotate_brt', '20180110_htl_glc_sc14_mmzp_rotate_brt', '20180110_htl_glc_sc15_mmzm_rotate_brt', '20180110_htl_glc_sc6_mmzm_rotate_brt', '20180112_htlglc_tl_sc11_mmzp_rotate_brt', '20180112_htlglc_tl_sc4_resille_rotate_brt'])



.. code:: ipython3

    import tqdm

Import images for each vector field object

.. code:: ipython3

    for f in vf.keys():
        vf[f].add_image_data(os.path.join('../data',vf[f].name+'.tif'))

Pick start points for each image

.. code:: ipython3

    L = []
    for f in vf.keys():
        L.append(vf[f].pick_start_points())



.. raw:: html

    
    <script src="http://localhost:51751/autoload.js?bokeh-autoload-element=1003&bokeh-absolute-url=http://localhost:51751&resources=none" id="1003"></script>



.. raw:: html

    
    <script src="http://localhost:51752/autoload.js?bokeh-autoload-element=1005&bokeh-absolute-url=http://localhost:51752&resources=none" id="1005"></script>



.. raw:: html

    
    <script src="http://localhost:51753/autoload.js?bokeh-autoload-element=1007&bokeh-absolute-url=http://localhost:51753&resources=none" id="1007"></script>



.. raw:: html

    
    <script src="http://localhost:51754/autoload.js?bokeh-autoload-element=1009&bokeh-absolute-url=http://localhost:51754&resources=none" id="1009"></script>



.. raw:: html

    
    <script src="http://localhost:51756/autoload.js?bokeh-autoload-element=1011&bokeh-absolute-url=http://localhost:51756&resources=none" id="1011"></script>



.. raw:: html

    
    <script src="http://localhost:51757/autoload.js?bokeh-autoload-element=1013&bokeh-absolute-url=http://localhost:51757&resources=none" id="1013"></script>



.. raw:: html

    
    <script src="http://localhost:51758/autoload.js?bokeh-autoload-element=1015&bokeh-absolute-url=http://localhost:51758&resources=none" id="1015"></script>



.. raw:: html

    
    <script src="http://localhost:51759/autoload.js?bokeh-autoload-element=1017&bokeh-absolute-url=http://localhost:51759&resources=none" id="1017"></script>



.. raw:: html

    
    <script src="http://localhost:51762/autoload.js?bokeh-autoload-element=1019&bokeh-absolute-url=http://localhost:51762&resources=none" id="1019"></script>



.. raw:: html

    
    <script src="http://localhost:51767/autoload.js?bokeh-autoload-element=1021&bokeh-absolute-url=http://localhost:51767&resources=none" id="1021"></script>


Save points from each plot object

.. code:: ipython3

    for i,f in enumerate(vf.keys()):
        vf[f].save_start_points(L[i])

Try calculating tracks with a guess of the time step

.. code:: ipython3

    for f in vf.keys():
        vf[f].calc_track_set(vf[f].starts,60,name='dt60')


.. parsed-literal::

    100%|██████████| 4/4 [00:00<00:00, 76.25it/s]
    100%|██████████| 4/4 [00:00<00:00, 92.83it/s]
    100%|██████████| 4/4 [00:00<00:00, 114.97it/s]
    100%|██████████| 8/8 [00:00<00:00, 93.37it/s]
    100%|██████████| 4/4 [00:00<00:00, 155.56it/s]
    100%|██████████| 4/4 [00:00<00:00, 117.00it/s]
    100%|██████████| 4/4 [00:00<00:00, 133.31it/s]
    100%|██████████| 4/4 [00:00<00:00, 96.07it/s]
    100%|██████████| 4/4 [00:00<00:00, 85.14it/s]
    100%|██████████| 4/4 [00:00<00:00, 93.33it/s]


Compile track dataframes

.. code:: ipython3

    Ldf = []
    for f in vf.keys():
        Ldf.append(vf[f].tracks)

.. code:: ipython3

    tracks = pd.concat(Ldf,keys=fs)

.. code:: ipython3

    tracks = tracks[tracks['name']=='dt60'].reset_index(
                                        ).drop(columns=['level_1']
                                        ).rename(columns={'level_0':'f'})
    tracks




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
          <th>x</th>
          <th>y</th>
          <th>t</th>
          <th>track</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>0</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>1</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1088.771472</td>
          <td>573.050488</td>
          <td>2</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1070.579972</td>
          <td>544.532344</td>
          <td>3</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1075.979244</td>
          <td>528.820894</td>
          <td>4</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>5</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1076.405471</td>
          <td>507.046099</td>
          <td>5</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>6</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1072.378276</td>
          <td>498.637164</td>
          <td>6</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1252.690540</td>
          <td>626.130975</td>
          <td>7</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>8</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1251.616616</td>
          <td>2542.678403</td>
          <td>8</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1262.913416</td>
          <td>2542.616285</td>
          <td>9</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>10</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1246.928816</td>
          <td>2542.824077</td>
          <td>10</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>11</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1239.878816</td>
          <td>2543.054021</td>
          <td>11</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>12</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1235.753910</td>
          <td>2542.957526</td>
          <td>12</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>13</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1231.134371</td>
          <td>2542.897567</td>
          <td>13</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>14</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1245.210185</td>
          <td>2542.829703</td>
          <td>14</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>15</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.682408</td>
          <td>2542.597897</td>
          <td>15</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>16</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1260.955008</td>
          <td>2542.457203</td>
          <td>16</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>17</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.653428</td>
          <td>2542.525285</td>
          <td>17</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>18</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.986628</td>
          <td>2542.695559</td>
          <td>18</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>19</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.657456</td>
          <td>2542.710772</td>
          <td>19</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>20</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.123266</td>
          <td>2542.573438</td>
          <td>20</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>21</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.697472</td>
          <td>2542.379908</td>
          <td>21</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>22</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.737352</td>
          <td>2542.625224</td>
          <td>22</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>23</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.753024</td>
          <td>2542.536004</td>
          <td>23</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>24</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.701190</td>
          <td>2542.513738</td>
          <td>24</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>25</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.696070</td>
          <td>2542.686118</td>
          <td>25</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>26</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.781696</td>
          <td>2542.783162</td>
          <td>26</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>27</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.972892</td>
          <td>2542.944814</td>
          <td>27</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>28</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.515808</td>
          <td>2543.150386</td>
          <td>28</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>29</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.229088</td>
          <td>2543.423446</td>
          <td>29</td>
          <td>0</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>7274</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.451802</td>
          <td>269.362333</td>
          <td>136</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7275</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.518153</td>
          <td>269.530447</td>
          <td>137</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7276</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.541718</td>
          <td>269.553814</td>
          <td>138</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7277</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.560226</td>
          <td>269.627099</td>
          <td>139</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7278</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.495677</td>
          <td>269.547292</td>
          <td>140</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7279</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.521217</td>
          <td>269.725623</td>
          <td>141</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7280</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.396584</td>
          <td>269.450421</td>
          <td>142</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7281</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.360714</td>
          <td>269.492079</td>
          <td>143</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7282</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.337381</td>
          <td>269.501165</td>
          <td>144</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7283</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.451429</td>
          <td>269.427124</td>
          <td>145</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7284</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.448977</td>
          <td>269.310078</td>
          <td>146</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7285</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.438550</td>
          <td>269.233108</td>
          <td>147</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7286</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.471803</td>
          <td>269.299951</td>
          <td>148</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7287</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.569092</td>
          <td>269.660966</td>
          <td>149</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7288</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.596273</td>
          <td>269.771042</td>
          <td>150</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7289</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.467637</td>
          <td>269.545540</td>
          <td>151</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7290</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.454084</td>
          <td>269.648345</td>
          <td>152</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7291</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.435269</td>
          <td>269.481833</td>
          <td>153</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7292</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.456330</td>
          <td>269.511120</td>
          <td>154</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7293</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.580469</td>
          <td>269.621619</td>
          <td>155</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7294</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.515572</td>
          <td>269.356312</td>
          <td>156</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7295</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.631431</td>
          <td>269.551713</td>
          <td>157</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7296</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.560893</td>
          <td>269.400476</td>
          <td>158</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7297</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.633023</td>
          <td>269.682078</td>
          <td>159</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7298</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.694824</td>
          <td>269.870035</td>
          <td>160</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7299</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.921197</td>
          <td>270.285350</td>
          <td>161</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7300</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.803108</td>
          <td>270.114968</td>
          <td>162</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7301</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.409202</td>
          <td>269.496611</td>
          <td>163</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7302</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.481978</td>
          <td>269.416020</td>
          <td>164</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
        <tr>
          <th>7303</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.520913</td>
          <td>269.406949</td>
          <td>165</td>
          <td>3</td>
          <td>dt60</td>
        </tr>
      </tbody>
    </table>
    <p>7304 rows × 6 columns</p>
    </div>



.. code:: ipython3

    import bokeh.plotting

.. code:: ipython3

    p = bokeh.plotting.figure(width=400,height=300)
    p.scatter?



.. parsed-literal::

    [0;31mSignature:[0m [0mp[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0m
    [0;31mDocstring:[0m
    Creates a scatter plot of the given x and y items.
    
    Args:
        x (str or seq[float]) : values or field names of center x coordinates
    
        y (str or seq[float]) : values or field names of center y coordinates
    
        size (str or list[float]) : values or field names of sizes in screen units
    
        marker (str, or list[str]): values or field names of marker types
    
        color (color value, optional): shorthand to set both fill and line color
    
        source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source.
            An attempt will be made to convert the object to :class:`~bokeh.models.sources.ColumnDataSource`
            if needed. If none is supplied, one is created for the user automatically.
    
        **kwargs: :ref:`userguide_styling_line_properties` and :ref:`userguide_styling_fill_properties`
    
    Examples:
    
        >>> p.scatter([1,2,3],[4,5,6], marker="square", fill_color="red")
        >>> p.scatter("data1", "data2", marker="mtype", source=data_source, ...)
    
    .. note::
        When passing ``marker="circle"`` it is also possible to supply a
        ``radius`` value in data-space units. When configuring marker type
        from a data source column, *all* markers incuding circles may only
        be configured with ``size`` in screen units.
    [0;31mFile:[0m      ~/anaconda3/envs/python36/lib/python3.6/site-packages/bokeh/plotting/figure.py
    [0;31mType:[0m      method



Create numerical index for file number

.. code:: ipython3

    tracks['f'] = tracks.f.astype('category')
    tracks['findex'] = tracks.f.cat.codes

.. code:: ipython3

    fig,ax = plt.subplots(figsize=(10,8))
    ax.scatter(tracks[tracks.t==0].x,tracks[tracks.t==0].y,
               c=tracks[tracks.t==0].findex,s=20)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x1d859b0748>




.. image:: output_23_1.png


Calculate average position of start points from each embryo and shift to
(0,0)

.. code:: ipython3

    avgpos = tracks.groupby('f')[['x','y']].mean().rename(columns={'x':'xavg','y':'yavg'})
    avgpos




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
          <th>xavg</th>
          <th>yavg</th>
        </tr>
        <tr>
          <th>f</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>20180108_htl_glc_sc11_mmzm_rotate_brt</th>
          <td>1280.539090</td>
          <td>664.808244</td>
        </tr>
        <tr>
          <th>20180108_htl_glc_sc2_mmzm_wp_rotate_brt</th>
          <td>1250.991146</td>
          <td>683.164164</td>
        </tr>
        <tr>
          <th>20180108_htl_glc_sc9_mmzp_rotate_brt</th>
          <td>640.682175</td>
          <td>506.318645</td>
        </tr>
        <tr>
          <th>20180110_htl_glc-CreateImageSubset-01_sc10_wt_rotate_brt</th>
          <td>874.460592</td>
          <td>226.181494</td>
        </tr>
        <tr>
          <th>20180110_htl_glc-CreateImageSubset-02_sc11_htl_rotate_brt</th>
          <td>1059.138286</td>
          <td>443.533512</td>
        </tr>
        <tr>
          <th>20180110_htl_glc_sc14_mmzp_rotate_brt</th>
          <td>1173.814278</td>
          <td>534.076384</td>
        </tr>
        <tr>
          <th>20180110_htl_glc_sc15_mmzm_rotate_brt</th>
          <td>1036.934953</td>
          <td>47.678660</td>
        </tr>
        <tr>
          <th>20180110_htl_glc_sc6_mmzm_rotate_brt</th>
          <td>1079.749194</td>
          <td>620.550426</td>
        </tr>
        <tr>
          <th>20180112_htlglc_tl_sc11_mmzp_rotate_brt</th>
          <td>836.187739</td>
          <td>301.807815</td>
        </tr>
        <tr>
          <th>20180112_htlglc_tl_sc4_resille_rotate_brt</th>
          <td>1246.183802</td>
          <td>487.280176</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    tracks = tracks.join(avgpos,on='f')


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-78-28bad3d9371e> in <module>
    ----> 1 tracks = tracks.join(avgpos,on='f')
          2 tracks.head()


    ~/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/frame.py in join(self, other, on, how, lsuffix, rsuffix, sort)
       6334         # For SparseDataFrame's benefit
       6335         return self._join_compat(other, on=on, how=how, lsuffix=lsuffix,
    -> 6336                                  rsuffix=rsuffix, sort=sort)
       6337 
       6338     def _join_compat(self, other, on=None, how='left', lsuffix='', rsuffix='',


    ~/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/frame.py in _join_compat(self, other, on, how, lsuffix, rsuffix, sort)
       6349             return merge(self, other, left_on=on, how=how,
       6350                          left_index=on is None, right_index=True,
    -> 6351                          suffixes=(lsuffix, rsuffix), sort=sort)
       6352         else:
       6353             if on is not None:


    ~/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
         60                          copy=copy, indicator=indicator,
         61                          validate=validate)
    ---> 62     return op.get_result()
         63 
         64 


    ~/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/reshape/merge.py in get_result(self)
        572 
        573         llabels, rlabels = items_overlap_with_suffix(ldata.items, lsuf,
    --> 574                                                      rdata.items, rsuf)
        575 
        576         lindexers = {1: left_indexer} if left_indexer is not None else {}


    ~/anaconda3/envs/python36/lib/python3.6/site-packages/pandas/core/internals.py in items_overlap_with_suffix(left, lsuffix, right, rsuffix)
       5242         if not lsuffix and not rsuffix:
       5243             raise ValueError('columns overlap but no suffix specified: '
    -> 5244                              '{rename}'.format(rename=to_rename))
       5245 
       5246         def lrenamer(x):


    ValueError: columns overlap but no suffix specified: Index(['xavg', 'yavg'], dtype='object')


.. code:: ipython3

    tracks.head()




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
          <th>x</th>
          <th>y</th>
          <th>t</th>
          <th>track</th>
          <th>name</th>
          <th>findex</th>
          <th>xavg</th>
          <th>yavg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>0</td>
          <td>0</td>
          <td>dt60</td>
          <td>0</td>
          <td>1280.53909</td>
          <td>664.808244</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>1</td>
          <td>0</td>
          <td>dt60</td>
          <td>0</td>
          <td>1280.53909</td>
          <td>664.808244</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1088.771472</td>
          <td>573.050488</td>
          <td>2</td>
          <td>0</td>
          <td>dt60</td>
          <td>0</td>
          <td>1280.53909</td>
          <td>664.808244</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1070.579972</td>
          <td>544.532344</td>
          <td>3</td>
          <td>0</td>
          <td>dt60</td>
          <td>0</td>
          <td>1280.53909</td>
          <td>664.808244</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1075.979244</td>
          <td>528.820894</td>
          <td>4</td>
          <td>0</td>
          <td>dt60</td>
          <td>0</td>
          <td>1280.53909</td>
          <td>664.808244</td>
        </tr>
      </tbody>
    </table>
    </div>



Subtract average position from each x and y

.. code:: ipython3

    tracks['xpr'] = tracks['x'] - tracks['xavg']
    tracks['ypr'] = tracks['y'] - tracks['yavg']

Plot hopefully aligned positions

.. code:: ipython3

    fig,ax = plt.subplots(figsize=(10,8))
    ax.scatter(tracks.xpr,tracks.ypr,c=tracks.findex)




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x1d8d9da518>




.. image:: output_31_1.png


.. code:: ipython3

    tracks.track.unique()




.. parsed-literal::

    array([0, 1, 2, 3, 4, 5, 6, 7])



.. code:: ipython3

    tracks.set_index(['track','findex']).index.is_unique




.. parsed-literal::

    False



.. code:: ipython3

    tracks.set_index(['track','findex'])




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
          <th>f</th>
          <th>x</th>
          <th>y</th>
          <th>t</th>
          <th>name</th>
          <th>xavg</th>
          <th>yavg</th>
          <th>xpr</th>
          <th>ypr</th>
        </tr>
        <tr>
          <th>track</th>
          <th>findex</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="30" valign="top">0</th>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>0</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-179.623412</td>
          <td>-66.052575</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1100.915678</td>
          <td>598.755670</td>
          <td>1</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-179.623412</td>
          <td>-66.052575</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1088.771472</td>
          <td>573.050488</td>
          <td>2</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-191.767618</td>
          <td>-91.757756</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1070.579972</td>
          <td>544.532344</td>
          <td>3</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-209.959118</td>
          <td>-120.275900</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1075.979244</td>
          <td>528.820894</td>
          <td>4</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-204.559846</td>
          <td>-135.987350</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1076.405471</td>
          <td>507.046099</td>
          <td>5</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-204.133619</td>
          <td>-157.762145</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1072.378276</td>
          <td>498.637164</td>
          <td>6</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-208.160814</td>
          <td>-166.171080</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1252.690540</td>
          <td>626.130975</td>
          <td>7</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-27.848550</td>
          <td>-38.677269</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1251.616616</td>
          <td>2542.678403</td>
          <td>8</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-28.922474</td>
          <td>1877.870159</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1262.913416</td>
          <td>2542.616285</td>
          <td>9</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-17.625674</td>
          <td>1877.808041</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1246.928816</td>
          <td>2542.824077</td>
          <td>10</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-33.610274</td>
          <td>1878.015833</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1239.878816</td>
          <td>2543.054021</td>
          <td>11</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-40.660274</td>
          <td>1878.245777</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1235.753910</td>
          <td>2542.957526</td>
          <td>12</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-44.785180</td>
          <td>1878.149282</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1231.134371</td>
          <td>2542.897567</td>
          <td>13</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-49.404719</td>
          <td>1878.089323</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1245.210185</td>
          <td>2542.829703</td>
          <td>14</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-35.328905</td>
          <td>1878.021459</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.682408</td>
          <td>2542.597897</td>
          <td>15</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-20.856681</td>
          <td>1877.789653</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1260.955008</td>
          <td>2542.457203</td>
          <td>16</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-19.584081</td>
          <td>1877.648959</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.653428</td>
          <td>2542.525285</td>
          <td>17</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-20.885661</td>
          <td>1877.717041</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.986628</td>
          <td>2542.695559</td>
          <td>18</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.552461</td>
          <td>1877.887315</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.657456</td>
          <td>2542.710772</td>
          <td>19</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.881633</td>
          <td>1877.902528</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.123266</td>
          <td>2542.573438</td>
          <td>20</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.415823</td>
          <td>1877.765194</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.697472</td>
          <td>2542.379908</td>
          <td>21</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-21.841617</td>
          <td>1877.571664</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.737352</td>
          <td>2542.625224</td>
          <td>22</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.801737</td>
          <td>1877.816980</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.753024</td>
          <td>2542.536004</td>
          <td>23</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.786065</td>
          <td>1877.727760</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.701190</td>
          <td>2542.513738</td>
          <td>24</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.837899</td>
          <td>1877.705494</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.696070</td>
          <td>2542.686118</td>
          <td>25</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.843020</td>
          <td>1877.877874</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.781696</td>
          <td>2542.783162</td>
          <td>26</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.757394</td>
          <td>1877.974918</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1257.972892</td>
          <td>2542.944814</td>
          <td>27</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.566198</td>
          <td>1878.136570</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1258.515808</td>
          <td>2543.150386</td>
          <td>28</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-22.023282</td>
          <td>1878.342142</td>
        </tr>
        <tr>
          <th>0</th>
          <td>20180108_htl_glc_sc11_mmzm_rotate_brt</td>
          <td>1259.229088</td>
          <td>2543.423446</td>
          <td>29</td>
          <td>dt60</td>
          <td>1280.539090</td>
          <td>664.808244</td>
          <td>-21.310002</td>
          <td>1878.615202</td>
        </tr>
        <tr>
          <th>...</th>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th rowspan="30" valign="top">3</th>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.451802</td>
          <td>269.362333</td>
          <td>136</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.731999</td>
          <td>-217.917842</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.518153</td>
          <td>269.530447</td>
          <td>137</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.665649</td>
          <td>-217.749728</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.541718</td>
          <td>269.553814</td>
          <td>138</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.642083</td>
          <td>-217.726361</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.560226</td>
          <td>269.627099</td>
          <td>139</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.623576</td>
          <td>-217.653076</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.495677</td>
          <td>269.547292</td>
          <td>140</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.688125</td>
          <td>-217.732883</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.521217</td>
          <td>269.725623</td>
          <td>141</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.662584</td>
          <td>-217.554553</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.396584</td>
          <td>269.450421</td>
          <td>142</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.787218</td>
          <td>-217.829754</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.360714</td>
          <td>269.492079</td>
          <td>143</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.823087</td>
          <td>-217.788096</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.337381</td>
          <td>269.501165</td>
          <td>144</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.846421</td>
          <td>-217.779011</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.451429</td>
          <td>269.427124</td>
          <td>145</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.732373</td>
          <td>-217.853051</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.448977</td>
          <td>269.310078</td>
          <td>146</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.734824</td>
          <td>-217.970097</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.438550</td>
          <td>269.233108</td>
          <td>147</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.745252</td>
          <td>-218.047067</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.471803</td>
          <td>269.299951</td>
          <td>148</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.711999</td>
          <td>-217.980224</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.569092</td>
          <td>269.660966</td>
          <td>149</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.614710</td>
          <td>-217.619210</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.596273</td>
          <td>269.771042</td>
          <td>150</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.587528</td>
          <td>-217.509133</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.467637</td>
          <td>269.545540</td>
          <td>151</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.716165</td>
          <td>-217.734636</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.454084</td>
          <td>269.648345</td>
          <td>152</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.729718</td>
          <td>-217.631831</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.435269</td>
          <td>269.481833</td>
          <td>153</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.748533</td>
          <td>-217.798342</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.456330</td>
          <td>269.511120</td>
          <td>154</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.727472</td>
          <td>-217.769055</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.580469</td>
          <td>269.621619</td>
          <td>155</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.603333</td>
          <td>-217.658556</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.515572</td>
          <td>269.356312</td>
          <td>156</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.668230</td>
          <td>-217.923864</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.631431</td>
          <td>269.551713</td>
          <td>157</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.552371</td>
          <td>-217.728462</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.560893</td>
          <td>269.400476</td>
          <td>158</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.622909</td>
          <td>-217.879699</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.633023</td>
          <td>269.682078</td>
          <td>159</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.550779</td>
          <td>-217.598098</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.694824</td>
          <td>269.870035</td>
          <td>160</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.488977</td>
          <td>-217.410141</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.921197</td>
          <td>270.285350</td>
          <td>161</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.262605</td>
          <td>-216.994826</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.803108</td>
          <td>270.114968</td>
          <td>162</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.380693</td>
          <td>-217.165208</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.409202</td>
          <td>269.496611</td>
          <td>163</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.774600</td>
          <td>-217.783564</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.481978</td>
          <td>269.416020</td>
          <td>164</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.701823</td>
          <td>-217.864155</td>
        </tr>
        <tr>
          <th>9</th>
          <td>20180112_htlglc_tl_sc4_resille_rotate_brt</td>
          <td>1051.520913</td>
          <td>269.406949</td>
          <td>165</td>
          <td>dt60</td>
          <td>1246.183802</td>
          <td>487.280176</td>
          <td>-194.662888</td>
          <td>-217.873227</td>
        </tr>
      </tbody>
    </table>
    <p>7304 rows × 9 columns</p>
    </div>



Save tracks for later follow up

.. code:: ipython3

    tracks.to_csv('20181128-tracking.csv')
