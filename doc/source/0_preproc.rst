前置工作
========

安装依赖包 ::

   tensorflow>=2.7.0
   numpy
   pandas
   tqdm
   ipython

解压从Kaggle官网下载的 ``recsys-challenge-2015.zip`` ，并将 ``*.dat`` 文件放于 ``dataset`` 目录下的 ``yoochoose-data`` 文件夹中。

下载网址：https://www.kaggle.com/chadgostopp/recsys-challenge-2015

解压完成后， ``dataset`` 目录的结构应如下所示 ::

   dataset
   └── yoochoose-data
      ├── dataset-README.txt
      ├── yoochoose-buys.dat
      ├── yoochoose-clicks.dat
      └── yoochoose-test.dat