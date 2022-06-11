import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.layers import Input, Dense, Dropout, GRU


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""
    def __init__(self, data, sep='\t', session_key='SessionId', item_key='ItemId', time_key='Time', n_samples=-1, itemmap=None, time_sort=False):
      
        """!
       
        @param    path csv文件的路径
        @param    sep csv分隔符 
        @param    session_key 会话字段名称
        @param    item_key 对象字段名称
        @param    time_key  时间字段名称
        @param    n_samples 使用样本数，-1表示使用完整数据集
        @param    itemmap 对象ID值和索引的映射关系
        @param    time_sort 是否按时间排序会话
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
        """!
        @return 每个会话与第一个会话之间的偏移量
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets

    def order_session_idx(self):
        """!
        @brief 将会话排序
        @return 将会话排序的索引数组
        """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        return session_idx_arr

    def add_item_indices(self, itemmap=None):
        """!
        @brief 将对象索引列item_idx加入df中
        @param itemmap 对象ID值和索引的映射关系
       
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key:item_ids,
                                   'item_idx':item2idx[item_ids].values})

        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    @property
    def items(self):
        """!
        @brief 去重
        """            
        return self.itemmap.ItemId.unique()


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""
    def __init__(self, dataset, batch_size=50):
        """!
        @param dataset 生成切片的会话数据集
        @param batch_size 切片大小
        @param done_sessions_counter 已完成会话的个数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0

    def __iter__(self):
        """! 
        @return 并行训练切片的迭代器
            input (B,): 稍后将编码为一个热向量的项目索引
            target (B,): 存储目标项索引的变量
            masks: 指示要终止的会话位置的Numpy数组
        """

        df = self.dataset.df
        session_key='SessionId'
        item_key='ItemId'
        time_key='TimeStamp'
        self.n_items = df[item_key].nunique()+1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = [] # indicator for the sessions to be terminated
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


def create_model(args):
    """!
    创建模型
    @param args 参数
    @return 模型
    """    
    emb_size = 50
    hidden_units = 100
    size = emb_size

    inputs = Input(batch_shape=(args.batch_size, 1, args.train_n_items))
    gru, gru_states = GRU(hidden_units, stateful=True, return_state=True, name="GRU")(inputs)
    drop2 = Dropout(0.25)(gru)
    predictions = Dense(args.train_n_items, activation='softmax')(drop2)
    model = Model(inputs=inputs, outputs=[predictions])
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()

    filepath='./model_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    return model


def get_metrics(model, args, train_generator_map, recall_k=20, mrr_k=20):
    """!
    计算度量指标
    @param model 模型
    @param args 参数
    @param train_generator_map 训练生成器映射
    @param recall_k 召回率
    @param mrr_k 正确检索结果值在检索结果中的排名
    @return 度量指标
    """
    test_dataset = SessionDataset(args.test_data, itemmap=train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    n = 0
    rec_sum = 0
    mrr_sum = 0

    print("Evaluating model...")
    for feat, label, mask in test_generator:

        gru_layer = model.get_layer(name="GRU")
        hidden_states = gru_layer.states[0].numpy()
        for elt in mask:
            hidden_states[elt, :] = 0
        gru_layer.reset_states(states=hidden_states)

        target_oh = to_categorical(label, num_classes=args.train_n_items)
        input_oh  = to_categorical(feat,  num_classes=args.train_n_items)
        input_oh = np.expand_dims(input_oh, axis=1)

        pred = model.predict(input_oh, batch_size=args.batch_size)

        for row_idx in range(feat.shape[0]):
            pred_row = pred[row_idx]
            label_row = target_oh[row_idx]

            rec_idx =  pred_row.argsort()[-recall_k:][::-1]
            mrr_idx =  pred_row.argsort()[-mrr_k:][::-1]
            tru_idx = label_row.argsort()[-1:][::-1]

            n += 1

            if tru_idx[0] in rec_idx:
                rec_sum += 1

            if tru_idx[0] in mrr_idx:
                mrr_sum += 1/int((np.where(mrr_idx == tru_idx[0])[0]+1))

    recall = rec_sum/n
    mrr = mrr_sum/n
    return (recall, recall_k), (mrr, mrr_k)


def train_model(model, args):
    """!
    训练模型
    @param model 模型
    @param args 参数
    """    
    train_dataset = SessionDataset(args.train_data)
    model_to_train = model
    batch_size = args.batch_size

    for epoch in range(1, args.epochs):
        with tqdm(total=args.train_samples_qty) as pbar:
            loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            for feat, target, mask in loader:

                gru_layer = model_to_train.get_layer(name="GRU")
                hidden_states = gru_layer.states[0].numpy()
                for elt in mask:
                    hidden_states[elt, :] = 0
                gru_layer.reset_states(states=hidden_states)

                input_oh = to_categorical(feat, num_classes=loader.n_items)
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=loader.n_items)

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)

        if args.save_weights:
            print("Saving weights...")
            model_to_train.save('./GRU4REC_{}.h5'.format(epoch))

        if args.eval_all_epochs:
            (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)
            print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
            print("\t - MRR@{}    epoch {}: {:5f}\n".format(mrr_k, epoch, mrr))

    if not args.eval_all_epochs:
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)
        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, args.epochs, rec))
        print("\t - MRR@{}    epoch {}: {:5f}\n".format(mrr_k, args.epochs, mrr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras GRU4REC: session-based recommendations')
    parser.add_argument('--resume', type=str, help='stored model path to continue training')
    parser.add_argument('--train-path', type=str, default='../../processedData/rsc15_train_tr.txt')
    parser.add_argument('--eval-only', type=bool, default=False)
    parser.add_argument('--dev-path', type=str, default='../../processedData/rsc15_train_valid.txt')
    parser.add_argument('--test-path', type=str, default='../../processedData/rsc15_test.txt')
    parser.add_argument('--batch-size', type=str, default=512)
    parser.add_argument('--eval-all-epochs', type=bool, default=False)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    args.train_data = pd.read_csv(args.train_path, sep='\t', dtype={'ItemId': np.int64})
    args.dev_data   = pd.read_csv(args.dev_path,   sep='\t', dtype={'ItemId': np.int64})
    args.test_data  = pd.read_csv(args.test_path,  sep='\t', dtype={'ItemId': np.int64})

    args.train_n_items = len(args.train_data['ItemId'].unique()) + 1

    args.train_samples_qty = len(args.train_data['SessionId'].unique()) + 1
    args.test_samples_qty = len(args.test_data['SessionId'].unique()) + 1

    if args.resume:
        try:
            model = keras.models.load_model(args.resume)
            print("Model checkpoint '{}' loaded!".format(args.resume))
        except OSError:
            print("Model checkpoint could not be loaded. Training from scratch...")
            model = create_model(args)
    else:
        model = create_model(args)

    if args.eval_only:
        train_dataset = SessionDataset(args.train_data)
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model, args, train_dataset.itemmap)
        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, -1, rec))
        print("\t - MRR@{}    epoch {}: {:5f}\n".format(mrr_k, -1, mrr))
    else:
        train_model(model, args)
