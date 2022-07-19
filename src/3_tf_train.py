import argparse
import shutil
import gzip

import IPython
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.backend import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# consts from rsc15 dataset
# (46138, 512)
# n_items = 37484
test_samples_qty = 15325
train_samples_qty = 7953886

emb_size = 50
hidden_units = 32
size = emb_size
batch_size = 512
train_n_items = 37484
epochs = 3


class Model(tf.Module):

    def __init__(self):
        inputs = Input(shape=(1, train_n_items), batch_size=batch_size)
        init_states = Input(shape=(hidden_units,), batch_size=batch_size)
        gru, gru_states = GRU(hidden_units, stateful=False, return_state=True, name="GRU")(inputs, initial_state=init_states)

        drop2 = Dropout(0.25)(gru)
        predictions = Dense(train_n_items, activation='softmax')(drop2)

        self.model = tf.keras.Model(inputs=[inputs, init_states], outputs=[predictions, gru_states])
        self._OPTIM = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(
            optimizer=self._OPTIM,
        )
        self.model.summary()

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size, hidden_units], tf.float32),
        # tf.TensorSpec([batch_size], tf.int32),
    ])
    def train(self, feat, target, initstate): # mask):
        # diag = tf.linalg.diag(tf.cast(mask, tf.float32))
        # self.initstate = tf.matmul(diag, self.initstate)
        input_oh = tf.one_hot(feat, depth=train_n_items)
        input_oh = tf.expand_dims(input_oh, axis=1)
        input_oh = tf.cast(input_oh, tf.float32)

        target_oh = tf.one_hot(target, depth=train_n_items)
        target_oh = tf.cast(target_oh, tf.float32)

        with tf.GradientTape() as tape:
            prediction, state = self.model([input_oh, initstate])
            loss = tf.reduce_mean(categorical_crossentropy(target_oh, prediction))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {
            "loss": loss,
            "state": state,
        }

    @tf.function(input_signature=[
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size, hidden_units], tf.float32),
    ])
    def infer(self, feat, initstate):
        input_oh = tf.one_hot(feat, depth=train_n_items)
        input_oh = tf.expand_dims(input_oh, axis=1)
        input_oh = tf.cast(input_oh, tf.float32)
        output, state = self.model([input_oh, initstate])
        return {
            "output": output,
            "state": state,
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value()
                           for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

    # The `eval` function returns loss, recall and MRR.
    @tf.function(input_signature=[
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size, hidden_units], tf.float32),
        tf.TensorSpec([], tf.int32)
    ])
    def eval(self, feat, target, initstate, top_k):
        input_oh = tf.one_hot(feat, depth=train_n_items)
        input_oh = tf.expand_dims(input_oh, axis=1)
        input_oh = tf.cast(input_oh, tf.float32)

        target_oh = tf.one_hot(target, depth=train_n_items)
        target_oh = tf.cast(target_oh, tf.float32)

        prediction, state = self.model([input_oh, initstate])
        loss = tf.reduce_mean(categorical_crossentropy(target_oh, prediction))

        # pred_scores = tf.constant([prediction[i,x].numpy() for i,x in enumerate(target)])
        pred_scores = tf.gather_nd(prediction, tf.stack([tf.range(batch_size), target], axis=1))
        pred_scores = tf.expand_dims(pred_scores, axis=1)
        pred_rank = tf.math.greater_equal(prediction, pred_scores)
        pred_rank = tf.cast(pred_rank, tf.int32)
        pred_rank = tf.math.reduce_sum(pred_rank, axis=1)

        recall = tf.math.reduce_mean(tf.cast(tf.math.less_equal(pred_rank, top_k), tf.float32))
        mrr = tf.reduce_mean(tf.math.divide(1.0, tf.cast(pred_rank, tf.float32)))

        return {
            "loss": loss,
            "recall": recall,
            "mrr": mrr,
            "state": state,
        }

if __name__ == '__main__':
    # Export the TensorFlow model to the saved model
    m = Model()

    feats = np.load('feats_bs512.npy')
    targets = np.load('targets_bs512.npy')
    masks = np.load('masks_bs512.npy')
    total_num = feats.shape[0]

    assert feats.shape[0] == targets.shape[0] == masks.shape[0]

    ZERO = tf.zeros((batch_size, hidden_units), dtype=tf.float32)
    for epoch in range(epochs):
        last_state = tf.zeros((batch_size, hidden_units), dtype=tf.float32)
        with tqdm(total=total_num) as pbar:
            for i, (feat, mask, target) in enumerate(zip(feats, masks, targets)):
                if i % 5000 == 0:
                    result = m.eval(feat, target, ZERO, top_k=20)
                    result = {k: v.numpy() for k, v in result.items() if k != "state"}
                    print(f"[ {epoch} epochs, {i} steps ] {result}")

                mask = tf.cast(mask, tf.float32)
                diag = tf.linalg.diag(mask)
                last_state = tf.matmul(diag, last_state)
                result = m.train(
                    feat=feat,
                    target=target,
                    initstate=last_state,
                )
                loss = result['loss']
                last_state = result['state']

                pbar.set_description(f'Epoch: {i} Loss: {loss:.5f}')
                pbar.update()
        m.model.save_weights(f'weights/{hidden_units}-{epoch:03d}.h5')
        last_state = tf.zeros((batch_size, hidden_units), dtype=tf.float32)

    ####### convert SavedModel to TF Lite #########
    SAVED_MODEL_DIR = "saved_model"
    try:
        tf.saved_model.save(
            m,
            SAVED_MODEL_DIR,
            signatures={
                'train':
                    m.train.get_concrete_function(),
                'infer':
                    m.infer.get_concrete_function(),
                # 'save':
                #     m.save.get_concrete_function(),
                # 'restore':
                #     m.restore.get_concrete_function(),
                'eval':
                    m.eval.get_concrete_function(),
            })
    except Exception as e:
        print(e)
        IPython.embed()

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    tflite_model_gzip = gzip.compress(tflite_model)
    with open("saved_model.tflite", "wb") as f:
        f.write(tflite_model_gzip)
    shutil.rmtree(SAVED_MODEL_DIR)
