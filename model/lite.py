# https://www.tensorflow.org/lite/examples/on_device_training/overview#train_the_model

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, GRU

emb_size = 50
hidden_units = 100
size = emb_size
batch_size = 512
train_n_items = 37484

# consts from rsc15 dataset
# (46138, 512)
# n_items = 37484
test_samples_qty = 15325
train_samples_qty = 7953886


class Model(tf.Module):

    def __init__(self):
        inputs = Input(shape=(1, train_n_items), batch_size=batch_size)
        init_states = Input(shape=(hidden_units,), batch_size=batch_size)
        gru, gru_states = GRU(hidden_units, stateful=False, return_state=True, name="GRU")(inputs, initial_state=init_states)

        drop2 = Dropout(0.25)(gru)
        predictions = Dense(train_n_items, activation='softmax')(drop2)

        self.model = tf.keras.Model(inputs=[inputs, init_states], outputs=[predictions, gru_states])
        self._OPTIM = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self._LOSS_FN = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.model.compile(
            optimizer=self._OPTIM,
            loss=self._LOSS_FN,
            metrics=['accuracy'],
        )
        self.last_state = tf.zeros((batch_size, hidden_units))

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
        tf.TensorSpec([batch_size, 1, train_n_items], tf.int32),
        tf.TensorSpec([batch_size, train_n_items], tf.int32),
        tf.TensorSpec([batch_size], tf.int32),
    ])
    def train(self, feat, target, mask):
        diag = tf.linalg.diag(tf.cast(mask, tf.float32))
        self.last_state = tf.matmul(diag, self.last_state)
        target = tf.cast(target, tf.float32)
        with tf.GradientTape() as tape:
            prediction, self.last_state = self.model([feat, self.last_state])
            loss = self._LOSS_FN(prediction, target)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._OPTIM.apply_gradients(zip([gradients], [self.model.trainable_variables]))
        result = {"loss": loss}
        for grad in gradients:
            result[grad.name] = grad
            print(grad.name, grad.shape)
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([batch_size, 1, train_n_items], tf.int32),
        tf.TensorSpec([batch_size, hidden_units], tf.int32),
    ])
    def infer(self, feat, init_state):
        output, state = self.model([feat, init_state])
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


if __name__ == '__main__':
    # Export the TensorFlow model to the saved model
    SAVED_MODEL_DIR = "saved_model"
    m = Model()
    tf.saved_model.save(
        m,
        SAVED_MODEL_DIR,
        signatures={
            'train':
                m.train.get_concrete_function(),
            'infer':
                m.infer.get_concrete_function(),
            'save':
                m.save.get_concrete_function(),
            'restore':
                m.restore.get_concrete_function(),
        })

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()