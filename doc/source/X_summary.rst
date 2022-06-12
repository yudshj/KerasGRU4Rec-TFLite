


总结
========

该阶段最终生成文件包括TXT格式的数据集文件 ``dataset/final/*.txt`` 和TFLite Model二进制文件 ``src/saved_model.tflite`` ::

   KerasGRU4Rec-TFLite
   ├── dataset
   │   └── final
   │       ├── feats_bs512.txt
   │       ├── masks_bs512.txt
   │       └── targets_bs512.txt
   └── src
      └── saved_model.tflite