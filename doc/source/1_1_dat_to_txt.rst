获得纯文本格式的数据集
=========================

从 DAT 文件中提取文本数据
*************************

这段 Python 脚本从 ``yoochoose-clicks.dat`` 中提取数据并将其保存至 TXT 文件。

运行命令 ::

   python3 1_1_dat_to_txt.py

程序应该有如下输出 ::

   Full train set
         Events: 31637239
         Sessions: 7966257
         Items: 37483
   Test set
         Events: 71222
         Sessions: 15324
         Items: 6751
   Train set
         Events: 31579006
         Sessions: 7953885
         Items: 37483
   Validation set
         Events: 58233
         Sessions: 12372
         Items: 6359

运行前 ``dataset`` 目录结构 ::

   dataset
   └── yoochoose-data
       ├── dataset-README.txt
       ├── yoochoose-buys.dat
       ├── yoochoose-clicks.dat
       └── yoochoose-test.dat

运行后 ``dataset`` 目录结构 ::

   dataset
   ├── rsc15
   │   ├── rsc15_test.txt
   │   ├── rsc15_train_full.txt
   │   ├── rsc15_train_tr.txt
   │   └── rsc15_train_valid.txt
   └── yoochoose-data
       ├── dataset-README.txt
       ├── yoochoose-buys.dat
       ├── yoochoose-clicks.dat
       └── yoochoose-test.dat


dat\_to\_txt.py 代码说明
----------------------------

.. automodule:: 1_1_dat_to_txt
   :members:
   :undoc-members:
   :show-inheritance:

将提取的文本数据转为 TXT
******************************

需要借助类 :py:class:`ido_gru4rec.SessionDataLoader` 以及 :py:class:`ido_gru4rec.SessionDataset` 来实现。

在 ``src`` 目录下，使用命令 ::
   
   python3 1_2_get_txt_dataset.py \
     --train-path ../dataset/rsc15/rsc15_train_tr.txt \
     --dev-path ../dataset/rsc15/rsc15_train_valid.txt \
     --test-path ../dataset/rsc15/rsc15_test.txt

执行完命令后，在 ``dataset/final`` 目录下将生成三个文件： ``feats_bs512.txt`` 、 ``masks_bs512.txt`` 和 ``targets_bs512.txt`` 。

项目目录整体结构如下。 ::

   KerasGRU4Rec-TFLite
   ├── dataset
   │   ├── final
   │   │   ├── feats_bs512.txt
   │   │   ├── masks_bs512.txt
   │   │   └── targets_bs512.txt
   │   ├── README.md
   │   ├── recsys-challenge-2015.zip
   │   ├── rsc15
   │   │   ├── rsc15_test.txt
   │   │   ├── rsc15_train_full.txt
   │   │   ├── rsc15_train_tr.txt
   │   │   └── rsc15_train_valid.txt
   │   └── yoochoose-data
   │       ├── dataset-README.txt
   │       ├── yoochoose-buys.dat
   │       ├── yoochoose-clicks.dat
   │       └── yoochoose-test.dat
   ├── README.md
   ├── requirements.txt
   └── src
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

   5 directories, 25 files


get\_txt\_dataset.py 代码说明
------------------------------

.. automodule:: 1_2_get_txt_dataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __iter__