
import IPython
import numpy as np
import tensorflow as tf
from tqdm import tqdm


emb_size = 50
hidden_units = 100
size = emb_size
batch_size = 512
train_n_items = 37484
rsc15_dataset = np.load('feats_targets_masks_bs512.npy', allow_pickle=False)
feats = rsc15_dataset[0]
targets = rsc15_dataset[1]
masks = rsc15_dataset[2]
total_num = rsc15_dataset.shape[1]

with open('saved_model.tflite', 'rb') as f:
    tflite_model = f.read()
print(len(tflite_model))
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

train = interpreter.get_signature_runner("train")
infer = interpreter.get_signature_runner("infer")
save = interpreter.get_signature_runner("save")
restore = interpreter.get_signature_runner("restore")

last_state = tf.zeros((batch_size, hidden_units), dtype=tf.float32)

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
