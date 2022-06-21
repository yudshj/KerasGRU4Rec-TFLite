获得TFLite Model模型二进制文件
==============================

运行命令 ::

   python3 2_get_tflite_model.py

此时应在 ``src`` 文件夹下生成文件 ``saved_model.tflite`` 。项目根目录下的 ``src`` 目录结构如下。 ::

   src
   ├── 1_1_dat_to_txt.py
   ├── 1_2_get_txt_dataset.py
   ├── 2_get_tflite_model.py
   ├── 3_tflite_train.py
   ├── feats_bs512.npy
   ├── ido_gru4rec.py
   ├── masks_bs512.npy
   ├── origin_gru4rec.py
   ├── saved_model.tflite
   └── targets_bs512.npy

   0 directories, 10 files


get\_tflite\_model.py 代码说明
*****************

.. automodule:: 2_get_tflite_model
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__