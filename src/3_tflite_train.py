import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gzip

TFLITE_MODEL_PATH = 'saved_model.tflite'
"""保存的 TFLite 模型名称
"""

HIDDEN_UNITS = 100
"""隐藏层的单元数量
"""

BATCH_SIZE = 512
"""批次大小
"""


if __name__ == '__main__':
    feats = np.load('feats_bs512.npy')
    targets = np.load('targets_bs512.npy')
    masks = np.load('masks_bs512.npy')
    total_num = feats.shape[0]

    assert feats.shape[0] == targets.shape[0] == masks.shape[0]

    with open(TFLITE_MODEL_PATH, 'rb') as f:
        tflite_model = gzip.decompress(f.read())
    print(len(tflite_model))
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    train = interpreter.get_signature_runner("train")
    infer = interpreter.get_signature_runner("infer")
    save = interpreter.get_signature_runner("save")
    restore = interpreter.get_signature_runner("restore")

    last_state = tf.zeros((BATCH_SIZE, HIDDEN_UNITS), dtype=tf.float32)

    with tqdm(total=total_num) as pbar:
        for i, (feat, mask, target) in enumerate(zip(feats, masks, targets)):
            mask = tf.cast(mask, tf.float32)
            diag = tf.linalg.diag(mask)
            last_state = tf.matmul(diag, last_state)
            result = train(
                feat=feat,
                target=target,
                initstate=last_state,
            )
            loss = result['loss']
            last_state = result['state']

            pbar.set_description(f'Epoch: {i} Loss: {loss:.5f}')
            pbar.update()
