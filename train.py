import tensorflow as tf
import tiktoken
from scipy.spatial import distance
import os
import random
import numpy as np
import math
import os
from datetime import datetime
import time
import numpy as np
import json

# Default checkpoint data
checkpoint_data = {
    'last_epoch': 0,
    'other_data': None  # You can add other data if needed
}

if os.path.exists('checkpoint.txt'):
    with open('checkpoint.txt', 'r') as file:
        checkpoint_data = json.load(file)

last_epoch = checkpoint_data['last_epoch']


cfg = {
    'sequenceSize': 512,
    'dimension': 512,
    'arrayDimension': 8,
    'predictSteps': 8,
    'batchSize': 4096
}
learning_rate = 0.005

wandb_log = False  # disabled by default
wandb_project = "fast-model"
wandb_run_name = "run_combined" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=cfg, resume=True)

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, pad_size=None, depth=None, pool=None, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.depth = depth
        self.pool = pool
        self.pad_size = pad_size
        self.DEFAULT_KERNEL_INITIALIZER = "glorot_normal"
        
        if self.depth % self.num_heads != 0:
            raise ValueError(f"Error: depth({self.depth}) % numHead({self.num_heads}) != 0")

    def build(self, input_shape):
        input_feat_size = input_shape[-1]
        
        self.random_id_att = self.add_weight("random_id_att", shape=(1,), initializer=tf.initializers.Zeros())
        self.random_id_ff = self.add_weight("random_id_ff", shape=(1,), initializer=tf.initializers.Zeros())
        
        # Define other weights
        self.input_dense_weight = self.add_weight("input_dense_weight", shape=(input_feat_size, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.input_dense_bias = self.add_weight("input_dense_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.query_dense_weight = self.add_weight("query_dense_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.query_dense_bias = self.add_weight("query_dense_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.key_dense_weight = self.add_weight("key_dense_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.key_dense_bias = self.add_weight("key_dense_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.value_dense_weight = self.add_weight("value_dense_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.value_dense_bias = self.add_weight("value_dense_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.dense_weight = self.add_weight("dense_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.dense_bias = self.add_weight("dense_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.ff_dense1_weight = self.add_weight("ff_dense1_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.ff_dense1_bias = self.add_weight("ff_dense1_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.ff_dense2_weight = self.add_weight("ff_dense2_weight", shape=(self.depth, self.depth), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        self.ff_dense2_bias = self.add_weight("ff_dense2_bias", shape=(self.depth,), initializer=self.DEFAULT_KERNEL_INITIALIZER)
        
        self.built = True

    def call(self, inputs):
        K = tf
        print("Shape of inputs:", inputs.shape)
        batch_size = tf.shape(inputs)[0]
        
        flat_input = tf.reshape(inputs, [self.pad_size * batch_size, -1])
        flat_scaled_input = tf.matmul(flat_input, self.input_dense_weight) + self.input_dense_bias
        scaled_input = tf.reshape(flat_scaled_input, [batch_size, self.pad_size, -1])
        print("Shape of scaledInput:", scaled_input.shape)

        flat_query = tf.matmul(flat_scaled_input, self.query_dense_weight) + self.query_dense_bias
        flat_key = tf.matmul(flat_scaled_input, self.key_dense_weight) + self.key_dense_bias
        flat_value = tf.matmul(flat_scaled_input, self.value_dense_weight) + self.value_dense_bias

        query = tf.reshape(flat_query, [batch_size, self.pad_size, -1])
        key = tf.reshape(flat_key, [batch_size, self.pad_size, -1])
        value = tf.reshape(flat_value, [batch_size, self.pad_size, -1])

        query_t = tf.transpose(tf.reshape(query, [batch_size, -1, self.num_heads, self.depth // self.num_heads]), [0, 2, 1, 3])
        key_t = tf.transpose(tf.reshape(key, [batch_size, -1, self.num_heads, self.depth // self.num_heads]), [0, 2, 1, 3])
        value_t = tf.transpose(tf.reshape(value, [batch_size, -1, self.num_heads, self.depth // self.num_heads]), [0, 2, 1, 3])
        print("Shape of queryT:", query_t.shape)
        print("Shape of keyT:", key_t.shape)
        print("Shape of valueT:", value_t.shape)

        matmul_qk = tf.matmul(query_t, key_t, transpose_b=True)
        logits = matmul_qk / tf.sqrt(tf.cast(self.depth, tf.float32))
        print("Shape of logits before addition:", logits.shape)
        
        to_broadcast_mask = tf.ones([batch_size, self.num_heads, self.pad_size, self.pad_size])
        print("Shape of to_broadcast_mask:", to_broadcast_mask.shape)
        logits += (1.0 - to_broadcast_mask) * -1e9

        attention_weights = tf.nn.sigmoid(logits) * tf.nn.tanh(logits)
        scaled_attention = tf.matmul(attention_weights, value_t)
        
        scaled_attention_t = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention_t, [batch_size, -1, self.depth])
        flatten_concat_attention = tf.reshape(concat_attention, [batch_size * self.pad_size, -1])
        flatten_attention = tf.matmul(flatten_concat_attention, self.dense_weight) + self.dense_bias
        attention = tf.reshape(flatten_attention, [batch_size, self.pad_size, -1])
        print("Shape of attention:", attention.shape)

        normalized_latent = scaled_input + tf.squeeze(self.random_id_att) * attention
        flatten_normalized_latent = tf.reshape(normalized_latent, [batch_size * self.pad_size, -1])
        
        flat_ff1 = tf.matmul(flatten_normalized_latent, self.ff_dense1_weight) + self.ff_dense1_bias
        flat_rff1 = tf.nn.leaky_relu(flat_ff1)
        flat_ff2 = tf.matmul(flat_rff1, self.ff_dense2_weight) + self.ff_dense2_bias
        flat_dff2 = tf.nn.dropout(flat_ff2, 0.1)
        dff2 = tf.reshape(flat_dff2, [batch_size, self.pad_size, -1])

        output = normalized_latent + tf.squeeze(self.random_id_ff) * dff2

        print("Shape of output:", output.shape)
        output = tf.reshape(output, [batch_size, self.pad_size, self.depth])
        print("Shape of output after reshape:", output.shape)

        if self.pool:
            return tf.reduce_mean(output, axis=1)
        else:
            return output


    def compute_output_shape(self, input_shape):
        if self.pool:
            return (input_shape[0], self.depth)
        else:
            return (input_shape[0], input_shape[1], self.depth)

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config.update({
            'pool': self.pool,
            'pad_size': self.pad_size,
            'num_heads': self.num_heads,
            'depth': self.depth
        })
        return config

def addler(data, mod):
    MOD_ADLER = mod
    a, b = 1, 0

    for value in data:
        a = (a + value) % MOD_ADLER
        b = (b + a) % MOD_ADLER

    return b

words = {}
maxf = 0

# Get all files under './data' directory, excluding .DS_Store files
books = [os.path.join(root, file) for root, dirs, files in os.walk('./data') for file in files if not file.endswith('.DS_Store')]

# Shuffle the list
random.shuffle(books)
books

# Read the file
with open('./models/vec.vec', 'r') as f:
    v = f.readlines()

v.pop(0)  # Remove the first line
w2v = {}

for line in v:
    tokens = line.split()
    key = tokens.pop(0)
    vec = [float(token) for token in tokens if token]
    w2v[key] = vec

vw2v = list(w2v.items())
vw2v

# Initialize the encoding
tokenizer = tiktoken.encoding_for_model("gpt-4")

def estimate_mfu(model, fwdbwd_per_iter, dt):
    """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
    N = sum(np.prod(v.shape) for v in model.trainable_variables)
    
    L = cfg['sequenceSize']  # Number of layers
    H = cfg['dimension'] // cfg['sequenceSize']  # Number of heads
    Q = cfg['sequenceSize']  # Dimension per head
    T = cfg['sequenceSize']  # Max sequence length
    
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    
    # express our flops throughput as a ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    
    mfu = flops_achieved / flops_promised
    return mfu


def reparse(v):
    v = [int(reverser(item)) if len(item) else item for item in v]
    return tokenizer.decode(v)

import numpy as np

def reverser(arr):
    arr_np = np.array(arr)
    vectors = np.array([item[1] if item[1] else [0] * 8 for item in vw2v])
    distances = np.linalg.norm(vectors - arr_np, axis=1)
    min_index = np.argmin(distances)
    return vw2v[min_index][0]


# Assuming w2v is a dictionary defined globally
empty_vec = [0] * 8
def convert(n):
    vec = w2v.get(str(n), empty_vec)
    if vec == empty_vec:
        print(f"Token {n} not found in w2v dictionary!")
    return [float(i) for i in vec]

assert convert(11) != empty_vec

books = [file for file in os.listdir('./data') if file.endswith('.txt') and not file.startswith('.DS_Store')]
random.shuffle(books)

# test tokenizer. Ensure that all tokens are present in vec.vec
for book_path in books:
    full_path = os.path.join('./data', book_path)
    with open(full_path, 'r') as f:
        book = f.read()
    book = tokenizer.encode(book)

    for token in book:
        assert convert(token) != empty_vec, f"token {token} hasn't been found in w2v: {w2v}"

setx = []
last_batch = []
def run():
    input = tf.keras.layers.Input(shape=(cfg['sequenceSize'], cfg['predictSteps']))

    x = tf.keras.layers.Permute((2, 1))(input)
    print("Shape of x after permute:", x.shape)
    skip = x

    x = tf.keras.layers.Conv1D(filters=cfg['dimension'], kernel_size=1, strides=1, padding="same", activation="mish")(x)
    print("Shape of x after Conv1D:", x.shape)
    x = TransformerLayer(depth=cfg['dimension'], num_heads=8, pad_size=cfg['arrayDimension'])(x)
    print("Shape of x after first TransformerLayer:", x.shape)
    x = TransformerLayer(depth=cfg['sequenceSize'], num_heads=8, pad_size=cfg['arrayDimension'])(x)
    print("Shape of x after second TransformerLayer:", x.shape)

    x3a = tf.keras.layers.Permute((2, 1))(x)
    x1 = tf.keras.layers.Dense(units=cfg['arrayDimension'], activation="linear")(x3a)

    # Check if saved model exists
    if os.path.exists('./models/llm'):
        model = tf.keras.models.load_model('./models/llm')
        print("Loaded model from disk.")
    else:
        model = tf.keras.Model(inputs=[input], outputs=[x1])
        model.compile(loss=tf.keras.losses.Huber(), metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    model.summary()

    for book_path in books:
        full_path = os.path.join('./data', book_path)
        with open(full_path, 'r') as f:
            book = f.read()

        book = tokenizer.encode(book)

        def generator():
            global setx
            global last_batch
            setx = []
            n = 0
            for _ in range(len(book) // cfg['batchSize']):
                for k in range(cfg['batchSize']):
                    pool_slice = book[n + k: k + n + cfg['sequenceSize'] + cfg['predictSteps']]
                    # print(f'len pool_slise: {len(pool_slice)}')
                    # print(f'pool size first five elements: {pool_slice[:5]}')
                    xs = pool_slice[:cfg['sequenceSize']]
                    ys = pool_slice[-cfg['sequenceSize']:]

                    # print(f'xs shape: {np.array(xs).shape}, ys shape: {np.array(ys).shape}')
                    # print(f'xs start: {xs[:10]}, ys start: {ys[:10]}')
                    
                    xs_converted = np.array(list(map(convert, xs)), dtype=np.float32)
                    ys_converted = np.array(list(map(convert, ys)), dtype=np.float32)
                                            
                    setx.append([xs_converted, ys_converted])
                    # print(f'xs_converted: {xs_converted}, ys_converted: {ys_converted}')
                n += 1
                
                # Check if setx has enough data for a batch
                if len(setx) == cfg['batchSize']:
                    tx1 = np.array([item[0] for item in setx])
                    ty1 = np.array([item[1] for item in setx])
                    #print(f'tx1 shape: {tx1.shape}, ty1 shape: {ty1.shape}')
                    last_batch = list(setx)
                    setx = []
                    yield tx1, ty1



        dataset = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(cfg['batchSize'], cfg['sequenceSize'], cfg['predictSteps']), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg['batchSize'], cfg['sequenceSize'], cfg['predictSteps']), dtype=tf.float32)
        ))

        epoch_start_time = time.time()
        def on_epoch_end(epoch, logs):
            global epoch_start_time
            # Calculate dt - the time taken for the epoch
            dt = time.time() - time.epoch_start_time
            # Estimate MFU
            fwdbwd_per_iter = 1 # Assuming one forward and backward pass per iteration
            mfu = estimate_mfu(model, fwdbwd_per_iter, dt)

            if not last_batch:
                print("last_batch is empty, skipping this epoch end.")
                return
            model.save('./models/llm')

            checkpoint_data['last_epoch'] = last_epoch + epoch

            with open('checkpoint.txt', 'w') as file:
                json.dump(checkpoint_data, file)

            r = random.randint(0, len(last_batch) - 1)
            s = [last_batch[r]]
            tx11 = np.array([item[0] for item in s])
            res = model.predict(tx11)
            a = res[0]
            print('---------------------------------INPUT-----------------------------------------')
            print(reparse(last_batch[r][0]))
            print('---------------------------------REAL-----------------------------------------')
            print(reparse(last_batch[r][1]))
            print('--------------------------------PREDICT----------------------------------------')
            print(tokenizer.decode(list(map(lambda s: int(reverser(s)), a))))

            mfu = estimate_mfu(model, fwdbwd_per_iter, dt)

            if wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": last_epoch + epoch,
                            "tokens": (last_epoch + epoch) * len(book) // 128,
                            "loss/train": logs["loss"],
                            "accuracy": logs["accuracy"],
                            "lr": learning_rate,
                            "mfu": mfu * 100,  # convert to percentage
                        }
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")

        # num_batches_per_epoch = len(book) // cfg['batchSize']
        # steps_per_epoch=num_batches_per_epoch
        model.fit(dataset, epochs=len(book) // 128, 
                  callbacks=[
                      tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs: setattr(time, 'epoch_start_time', time.time()),
        on_epoch_end=on_epoch_end)
        ])

run()