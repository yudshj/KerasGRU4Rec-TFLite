import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

import tensorflow as tf


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data, sep='\t', session_key='SessionId', item_key='ItemId', time_key='Time', n_samples=-1, itemmap=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        self.add_item_indices(itemmap=itemmap)
        self.df.sort_values([session_key, time_key], inplace=True)

        # Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        # clicks within a session are next to each other, where the clicks within a session are time-ordered.

        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()

    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(
            self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[
                self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        return session_idx_arr

    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                   'item_idx': item2idx[item_ids].values})

        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap,
                           on=self.item_key, how='inner')

    @property
    def items(self):
        return self.itemmap.ItemId.unique()


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
            dataset (SessionDataset): the session dataset to generate the batches from
            batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        df = self.dataset.df
        session_key = 'SessionId'
        item_key = 'ItemId'
        time_key = 'TimeStamp'
        self.n_items = df[item_key].nunique()+1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices (for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                inp = idx_input
                target = idx_target
                yield inp, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]


def work(args):
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
