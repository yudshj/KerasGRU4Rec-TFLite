import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

import tensorflow as tf
from ido_gru4rec import SessionDataLoader, SessionDataset

def work(args):
    """模型训练以及保存

    :param args: 命令行参数
    :type args: argparse.Namespace
    """    
    train_dataset = SessionDataset(args.train_data)
    batch_size = args.batch_size

    loader = SessionDataLoader(train_dataset, batch_size=batch_size)
    feats = []
    targets = []
    masks = []
    for feat, target, mask in loader:
        feats.append(feat)
        targets.append(target)
        masks.append(mask)
    # # masks = [np.pad(mask, (0, batch_size - len(mask)), 'constant', constant_values=-1) for mask in masks]
    masks = [tf.reduce_max(tf.one_hot(mask, args.batch_size, dtype=tf.int32), axis=0) if len(
        mask) > 0 else tf.zeros(args.batch_size, dtype=tf.int32) for mask in masks]
    feats_arr = np.array(feats, dtype=np.int32)
    targets_arr = np.array(targets, dtype=np.int32)
    masks_arr = np.array(masks, dtype=np.int32)
    masks_arr = 1 - masks_arr
    np.save(f'feats_bs{batch_size}.npy', feats_arr)
    np.save(f'targets_bs{batch_size}.npy', targets_arr)
    np.save(f'masks_bs{batch_size}.npy', masks_arr)
    pathlib.Path('../dataset/final/').mkdir(parents=True, exist_ok=True)
    np.savetxt(f'../dataset/final/feats_bs{batch_size}.txt', feats_arr, delimiter=',', fmt='%d')
    np.savetxt(f'../dataset/final/targets_bs{batch_size}.txt', targets_arr, delimiter=',', fmt='%d')
    np.savetxt(f'../dataset/final/masks_bs{batch_size}.txt', masks_arr, delimiter=',', fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ido Keras GRU4REC: session-based recommendations')
    parser.add_argument('--train-path', type=str,
                        default='../../processedData/rsc15_train_tr.txt')
    parser.add_argument('--dev-path', type=str,
                        default='../../processedData/rsc15_train_valid.txt')
    parser.add_argument('--test-path', type=str,
                        default='../../processedData/rsc15_test.txt')
    parser.add_argument('--batch-size', type=str, default=512)
    args = parser.parse_args()

    args.train_data = pd.read_csv(
        args.train_path, sep='\t', dtype={'ItemId': np.int64})
    args.dev_data = pd.read_csv(
        args.dev_path,   sep='\t', dtype={'ItemId': np.int64})
    args.test_data = pd.read_csv(
        args.test_path,  sep='\t', dtype={'ItemId': np.int64})

    # args.train_n_items = len(args.train_data['ItemId'].unique()) + 1

    # args.train_samples_qty = len(args.train_data['SessionId'].unique()) + 1
    # args.test_samples_qty = len(args.test_data['SessionId'].unique()) + 1

    work(args)
