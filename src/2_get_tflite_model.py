# https://www.tensorflow.org/lite/examples/on_device_training/overview#train_the_model

import argparse
import IPython
from keras.backend import categorical_crossentropy

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, GRU
import shutil

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
    """兼容TFLite的GRU模型类。
    """
    def __init__(self):
        """初始化函数。Dropout率为0.25。
        """                
        super(Model, self).__init__()
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

    # “train”功能接收一批输入图像和标签
    @tf.function(input_signature=[
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size], tf.int32),
        tf.TensorSpec([batch_size, hidden_units], tf.float32),
        # tf.TensorSpec([batch_size], tf.int32),
    ])
    def train(self, feat, target, initstate): # mask):
        """在给出的训练数据上训练模型，并返回*损失函数的值*和*隐藏层状态*。

        :param feat: 特征向量。
        :type feat: tf.Tensor
        :param target: 待拟合的目标向量。
        :type target: tf.Tensor
        :param initstate: 预设的初始状态（通常为上一轮的*隐藏层状态*）。
        :type initstate: tf.Tensor
        :return: 损失值和状态
        :rtype: { "loss": float, "state": tf.Tensor }

        """
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
        """机器学习推理。返回*预测结果*和*隐藏层状态*。

        :param feat: 特征向量
        :type feat: tf.Tensor
        :param initstate: 初始状态
        :type initstate: tf.Tensor
        :return: *预测结果*和*隐藏层状态*
        :rtype: {"output": tf.Tensor, "state": tf.Tensor}
        """
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
        """该方法将创建检查点文件，并将模型保存到指定路径。

        :param checkpoint_path: 保存路径
        :type checkpoint_path: str
        :return: 保存路径字典
        :rtype: {"checkpoint_path": str}
        """

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        """该方法将加载检查点文件，并将模型加载到指定路径。

        :param checkpoint_path: 需要恢复的路径
        :type checkpoint_path: str
        :return: 模型Tensor
        :rtype: tf.Tensor
        """
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
    try:
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
    with open("saved_model.tflite", "wb") as f:
        f.write(tflite_model)
    shutil.rmtree(SAVED_MODEL_DIR)