#!/bin/bash
python3 1_1_dat_to_txt.py
python3 1_2_get_txt_dataset.py --train-path ../dataset/rsc15/rsc15_train_tr.txt --dev-path ../dataset/rsc15/rsc15_train_valid.txt --test-path ../dataset/rsc15/rsc15_test.txt
python3 2_get_tflite_model.py
python3 3_tf_train.py