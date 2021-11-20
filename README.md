- `src/origin_gru4rec.py` 原始 [KerasGRU4Rec](https://github.com/paxcema/KerasGRU4Rec)
- `src/gen_txt_dataset.py` 预处理 dataset 得到纯文本格式的数据集
- `src/ido_gru4rec.py` 修改后的模型，将 GRU inner state 作为模型输入
- `src/tfmodel_to_litemodel.py` 将 SavedModel 转换成 TFLiteModel
- `src/tflite_train.py` 使用 TFLite 的 Python API 训练 GRU 模型

---

预处理RSC15数据集，得到txt文件

```
python3 gen_txt_dataset.py --train-path ../dataset/rsc15/rsc15_train_tr.txt --dev-path ../dataset/rsc15/rsc15_train_valid.txt --test-path ../dataset/rsc15/rsc15_test.txt
```
