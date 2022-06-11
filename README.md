## 获得TFLite Model和TXT格式数据集

分为前置工作、获得纯文本格式的数据集、获得TFLite Model模型二进制文件、使用Python测试TFLite训练过程。

### 0. 前置工作

安装依赖包

```
tensorflow>=2.7.0
numpy
pandas
tqdm
ipython
```

解压从Kaggle官网（https://www.kaggle.com/chadgostopp/recsys-challenge-2015）下载的`recsys-challenge-2015.zip`，并将 `*.dat` 文件放于dataset目录下的`yoochoose-data`文件夹中。

```
dataset
├── recsys-challenge-2015.zip
└── yoochoose-data
    ├── dataset-README.txt
    ├── yoochoose-buys.dat
    ├── yoochoose-clicks.dat
    └── yoochoose-test.dat
```


### 1. 获得纯文本格式的数据集

==在`src`目录下==，运行`1_1_dat_to_txt.py`生成`rsc15_*.txt`。这一步耗时较长，需要约12分钟左右。

此时dataset目录下的文件夹应该如下。

```bash
dataset
├── recsys-challenge-2015.zip
├── rsc15
│   ├── rsc15_test.txt
│   ├── rsc15_train_full.txt
│   ├── rsc15_train_tr.txt
│   └── rsc15_train_valid.txt
└── yoochoose-data
    ├── dataset-README.txt
    ├── yoochoose-buys.dat
    ├── yoochoose-clicks.dat
    └── yoochoose-test.dat
```

程序输出应该如下.

```
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

```

==在`src`目录下==，使用命令 `python3 1_2_get_txt_dataset.py --train-path ../dataset/rsc15/rsc15_train_tr.txt --dev-path ../dataset/rsc15/rsc15_train_valid.txt --test-path ../dataset/rsc15/rsc15_test.txt` .

因为处理过程需要用到TensorFlow库，可能的输出如下.

```
2022-06-11 11:07:08.657191: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-06-11 11:07:09.906136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9471 MB memory:  -> device: 0, name: Tesla M40, pci bus id: 0000:08:00.0, compute capability: 5.2
2022-06-11 11:07:09.907472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 10723 MB memory:  -> device: 1, name: Tesla M40, pci bus id: 0000:09:00.0, compute capability: 5.2
```

此时在`dataset/final`目录下会生成三个文件。项目目录结构如下。

```
KerasGRU4Rec-TFLite
├── dataset
│   ├── final
│   │   ├── feats_bs512.txt
│   │   ├── masks_bs512.txt
│   │   └── targets_bs512.txt
│   ├── README.md
│   ├── recsys-challenge-2015.zip
│   ├── rsc15
│   │   ├── rsc15_test.txt
│   │   ├── rsc15_train_full.txt
│   │   ├── rsc15_train_tr.txt
│   │   └── rsc15_train_valid.txt
│   └── yoochoose-data
│       ├── dataset-README.txt
│       ├── yoochoose-buys.dat
│       ├── yoochoose-clicks.dat
│       └── yoochoose-test.dat
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
```

### 2. 获得TFLite Model模型二进制文件

==在`src`目录下==，使用命令 `python3 2_get_tflite_model.py`

因为处理过程需要用到TensorFlow库，可能的输出如下.

```
2022-06-11 11:10:35.291923: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-06-11 11:10:36.552777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9471 MB memory:  -> device: 0, name: Tesla M40, pci bus id: 0000:08:00.0, compute capability: 5.2
2022-06-11 11:10:36.554023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 10723 MB memory:  -> device: 1, name: Tesla M40, pci bus id: 0000:09:00.0, compute capability: 5.2
WARNING:absl:Found untraced functions such as gru_cell_layer_call_fn, gru_cell_layer_call_and_return_conditional_losses while saving (showing 2 of 2). These functions will not be directly callable after loading.
WARNING:absl:Importing a function (__inference_internal_grad_fn_8667) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
WARNING:absl:Importing a function (__inference_internal_grad_fn_8701) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
2022-06-11 11:10:45.787264: W tensorflow/core/common_runtime/graph_constructor.cc:805] Node 'while' has 11 outputs but the _output_shapes attribute specifies shapes for 21 outputs. Output shapes may be inaccurate.
2022-06-11 11:10:46.369879: W tensorflow/core/common_runtime/graph_constructor.cc:805] Node 'while' has 11 outputs but the _output_shapes attribute specifies shapes for 21 outputs. Output shapes may be inaccurate.
2022-06-11 11:10:47.384360: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:362] Ignored output_format.
2022-06-11 11:10:47.384407: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:365] Ignored drop_control_dependency.
2022-06-11 11:10:47.385437: I tensorflow/cc/saved_model/reader.cc:43] Reading SavedModel from: saved_model
2022-06-11 11:10:47.400664: I tensorflow/cc/saved_model/reader.cc:81] Reading meta graph with tags { serve }
2022-06-11 11:10:47.400714: I tensorflow/cc/saved_model/reader.cc:122] Reading SavedModel debug info (if present) from: saved_model
2022-06-11 11:10:47.440382: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
2022-06-11 11:10:47.449713: I tensorflow/cc/saved_model/loader.cc:228] Restoring SavedModel bundle.
2022-06-11 11:10:47.803252: I tensorflow/cc/saved_model/loader.cc:212] Running initialization op on SavedModel bundle at path: saved_model
2022-06-11 11:10:47.975020: I tensorflow/cc/saved_model/loader.cc:301] SavedModel load for tags { serve }; Status: success: OK. Took 589593 microseconds.
2022-06-11 11:10:48.300086: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:263] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2022-06-11 11:10:49.370983: W tensorflow/compiler/mlir/lite/flatbuffer_export.cc:1901] TFLite interpreter needs to link Flex delegate in order to run the model since it contains the following Select TFop(s):
Flex ops: FlexAddN, FlexBiasAddGrad, FlexCast, FlexEmptyTensorList, FlexRestore, FlexSave, FlexSigmoidGrad, FlexStridedSliceGrad, FlexTensorListFromTensor, FlexTensorListGetItem, FlexTensorListLength, FlexTensorListPopBack, FlexTensorListPushBack, FlexTensorListReserve, FlexTensorListSetItem, FlexTensorListStack, FlexZerosLike
Details:
        tf.AddN(tensor<!tf_type.variant<tensor<512x37484xf32>>>, tensor<!tf_type.variant<tensor<*xf32>>>) -> (tensor<!tf_type.variant>) : {_class = ["loc:@gradients/while_grad/gradients/grad_ys_2"], device = ""}
        tf.BiasAddGrad(tensor<*xf32>) -> (tensor<?xf32>) : {data_format = "NHWC", device = ""}
        tf.Cast(tensor<!tf_type.variant>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {Truncate = false}
        tf.Cast(tensor<*x!tf_type.variant>) -> (tensor<!tf_type.variant<tensor<2xi32>>>) : {Truncate = false}
        tf.Cast(tensor<*x!tf_type.variant>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {Truncate = false}
        tf.Cast(tensor<*x!tf_type.variant>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {Truncate = false}
        tf.Cast(tensor<*x!tf_type.variant>) -> (tensor<!tf_type.variant<tensor<i32>>>) : {Truncate = false}
        tf.EmptyTensorList(tensor<0xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<i32>>>) : {device = ""}
        tf.EmptyTensorList(tensor<1xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<2xi32>>>) : {device = ""}
        tf.EmptyTensorList(tensor<2xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {device = ""}
        tf.EmptyTensorList(tensor<2xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {device = ""}
        tf.Restore(tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<100x300xf32>) : {device = "", preferred_shard = -1 : i64}
        tf.Restore(tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<100x37484xf32>) : {device = "", preferred_shard = -1 : i64}
        tf.Restore(tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<2x300xf32>) : {device = "", preferred_shard = -1 : i64}
        tf.Restore(tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<37484x300xf32>) : {device = "", preferred_shard = -1 : i64}
        tf.Restore(tensor<!tf_type.string>, tensor<!tf_type.string>) -> (tensor<37484xf32>) : {device = "", preferred_shard = -1 : i64}
        tf.Save(tensor<!tf_type.string>, tensor<5x!tf_type.string>, tensor<37484x300xf32>, tensor<100x300xf32>, tensor<2x300xf32>, tensor<100x37484xf32>, tensor<37484xf32>) -> () : {device = ""}
        tf.SigmoidGrad(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>) : {device = ""}
        tf.StridedSliceGrad(tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<512x100xf32>) -> (tensor<1x512x100xf32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64}
        tf.TensorListFromTensor(tensor<1x512x100xf32>, tensor<2xi32>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {device = ""}
        tf.TensorListFromTensor(tensor<1x512x37484xf32>, tensor<2xi32>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {device = ""}
        tf.TensorListGetItem(tensor<!tf_type.variant<tensor<512x100xf32>>>, tensor<i32>, tensor<2xi32>) -> (tensor<512x100xf32>) : {device = ""}
        tf.TensorListGetItem(tensor<!tf_type.variant<tensor<512x37484xf32>>>, tensor<i32>, tensor<2xi32>) -> (tensor<512x37484xf32>) : {device = ""}
        tf.TensorListLength(tensor<!tf_type.variant<tensor<512x37484xf32>>>) -> (tensor<i32>) : {device = ""}
        tf.TensorListPopBack(tensor<!tf_type.variant<tensor<2xi32>>>, tensor<i32>) -> (tensor<*x!tf_type.variant>, tensor<*xi32>) : {device = ""}
        tf.TensorListPopBack(tensor<!tf_type.variant<tensor<512x100xf32>>>, tensor<i32>) -> (tensor<*x!tf_type.variant>, tensor<*xf32>) : {device = ""}
        tf.TensorListPopBack(tensor<!tf_type.variant<tensor<512x37484xf32>>>, tensor<i32>) -> (tensor<*x!tf_type.variant>, tensor<*xf32>) : {device = ""}
        tf.TensorListPopBack(tensor<!tf_type.variant<tensor<i32>>>, tensor<i32>) -> (tensor<*x!tf_type.variant>, tensor<*xi32>) : {device = ""}
        tf.TensorListPushBack(tensor<!tf_type.variant<tensor<2xi32>>>, tensor<2xi32>) -> (tensor<!tf_type.variant<tensor<2xi32>>>) : {device = ""}
        tf.TensorListPushBack(tensor<!tf_type.variant<tensor<512x100xf32>>>, tensor<512x100xf32>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {device = ""}
        tf.TensorListPushBack(tensor<!tf_type.variant<tensor<512x37484xf32>>>, tensor<512x37484xf32>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {device = ""}
        tf.TensorListPushBack(tensor<!tf_type.variant<tensor<i32>>>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<i32>>>) : {device = ""}
        tf.TensorListReserve(tensor<*xi32>, tensor<*xi32>) -> (tensor<!tf_type.variant<tensor<*xf32>>>) : {device = ""}
        tf.TensorListReserve(tensor<2xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {device = ""}
        tf.TensorListSetItem(tensor<!tf_type.variant<tensor<*xf32>>>, tensor<*xi32>, tensor<?x37484xf32>) -> (tensor<!tf_type.variant<tensor<*xf32>>>) : {device = ""}
        tf.TensorListSetItem(tensor<!tf_type.variant<tensor<512x100xf32>>>, tensor<i32>, tensor<512x100xf32>) -> (tensor<!tf_type.variant<tensor<512x100xf32>>>) : {device = ""}
        tf.TensorListStack(tensor<!tf_type.variant<tensor<512x100xf32>>>, tensor<2xi32>) -> (tensor<1x512x100xf32>) : {device = "", num_elements = 1 : i64}
        tf.ZerosLike(tensor<!tf_type.variant<tensor<512x37484xf32>>>) -> (tensor<!tf_type.variant<tensor<512x37484xf32>>>) : {device = ""}
See instructions: https://www.tensorflow.org/lite/guide/ops_select
```

关于模型转换相关的文档、以及已知的局限性，请参阅 https://www.tensorflow.org/lite/guide/ops_select 

此时应在src文件夹下生成文件`saved_model.tflite`。项目目录结构如下。

```
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
```

### 3. 使用Python测试TFLite训练过程

`python3 3_tflite_train.py`

这一步是为了确保步骤2中获得的二进制文件可用。

## 最终生成文件

包括TXT格式的数据集文件（`dataset/final/*.txt`）和TFLite Model二进制文件（`src/saved_model.tflite`）

```
KerasGRU4Rec-TFLite
├── dataset
│   └── final
│       ├── feats_bs512.txt
│       ├── masks_bs512.txt
│       └── targets_bs512.txt
└── src
    └── saved_model.tflite
```