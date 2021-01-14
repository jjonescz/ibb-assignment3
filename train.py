# %%
import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
# Parameters
DATASET_PATH = "data"  # path to extracted http://awe.fri.uni-lj.si/downloads/AWEDataset.zip
BATCH_SIZE = 64
SHUFFLE_SIZE = 500
IMAGE_W = 128
IMAGE_H = 128
IMAGE_C = 3
N_LABELS = 100
EPOCHS = [35, 35]
LEARNING_RATES = [-1e-3, -1e-4]  # negative value freezes EfficientNet
HIDDEN_LAYERS = [512, 512]
GROUP_NORM = 16
DROPOUT = 0.5
EXP_ID = "model-a"  # subfolder inside `out/` with saved state
AUGMENTATIONS = False
OUT_DIR = os.path.join("out", EXP_ID)

# %%
# Copy this file to output directory.
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copyfile(__file__, os.path.join(OUT_DIR, os.path.basename(__file__)))

# %%
# Load image paths.
translations = pd.read_csv('awe-translation.csv')
images = dict()
labels = dict()
for dataset in ['train', 'test']:
    rows = translations['AWE-Full image path'].str.startswith(dataset)
    filenames = translations[rows]['AWE image path']
    paths = map(lambda f: os.path.join(DATASET_PATH, f), filenames)
    images[dataset] = list(paths)
    labels[dataset] = list(translations[rows]['Subject ID'] - 1)

# %%
# Load images as Tensorflow Datasets.
datasets = dict()
for dataset in ['train', 'test']:
    def transform(image, label):
        # Load image from given path.
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=IMAGE_C)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((images[dataset], labels[dataset]))
    datasets[dataset] = ds.map(transform)

# %%
# Determine image shapes.
image_shapes = np.array([image.shape for image, _ in datasets['train']])
print(f'Min image size: {image_shapes.min(axis=0)}')
print(f'Max image size: {image_shapes.max(axis=0)}')
print(f'Avg image size: {image_shapes.mean(axis=0)}')

# %%
# Resize images.
for dataset, ds in datasets.items():
    def transform(image, label):
        image = tf.image.resize(image, (IMAGE_H, IMAGE_W))
        return image, label
    datasets[dataset] = ds.map(transform)

# %%
# Augment training images.
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_H + 20, IMAGE_W + 20)
    image = tf.image.resize(image, [tf.random.uniform([], minval=IMAGE_H, maxval=IMAGE_H + 40, dtype=tf.int32),
                                    tf.random.uniform([], minval=IMAGE_W, maxval=IMAGE_W + 40, dtype=tf.int32)])
    image = tf.image.random_crop(image, [IMAGE_H, IMAGE_W, IMAGE_C])
    return image, label
if AUGMENTATIONS:
    datasets['train'] = datasets['train'].map(augment).prefetch(tf.data.experimental.AUTOTUNE)

# %%
# Plot some images.
for i, (image, _) in enumerate(datasets['train']):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(image.numpy().astype('uint8'))
    if i == 8:
        break

# %%
# Split into training, validation and testing datasets.
ds_train_shuffled = datasets['train'].shuffle(SHUFFLE_SIZE)
ds_train = ds_train_shuffled.take(500).cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
ds_val = ds_train_shuffled.skip(500).cache().batch(BATCH_SIZE)
ds_test = datasets['test'].cache().batch(BATCH_SIZE)

# %%
# Plot class distribution.
def plot_class_distribution(ds, name):
    labels = [label.numpy() for _, label in ds.unbatch()]
    plt.hist(labels, bins=N_LABELS)
    plt.title(f'Class distribution in {name} data')
    plt.show()
plot_class_distribution(ds_train_shuffled.batch(BATCH_SIZE), 'original train')
plot_class_distribution(ds_train, 'train')
plot_class_distribution(ds_val, 'validation')
plot_class_distribution(ds_test, 'test')

# %%
# Load (or download) EfficientNet.
efficientnet = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(IMAGE_H, IMAGE_W, IMAGE_C))

# %%
# Construct CNN model.
x = inputs = tf.keras.layers.Input(shape=[IMAGE_H, IMAGE_W, IMAGE_C])
x = efficientnet(x)
x = tf.keras.layers.GlobalMaxPool2D()(x)
for h in HIDDEN_LAYERS:
    x = tf.keras.layers.Dense(h, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
x = tf.keras.layers.Dense(N_LABELS, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=x)

# %%
# Create callback which will save checkpoints during training.
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUT_DIR, 'train-{epoch:04d}.ckpt'),
    save_freq=5 * len(ds_train),  # save every 5th epoch
    save_weights_only=True,
    verbose=1
)

# %%
# Train.
epochs = 0
training = []
for e, lr in zip(EPOCHS, LEARNING_RATES):
    efficientnet.trainable = lr > 0
    model.compile(
        optimizer=tf.optimizers.Adam(abs(lr)),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    t = model.fit(
        ds_train,
        initial_epoch=epochs,
        epochs=epochs + e,
        validation_data=ds_val,
        callbacks=[cp_callback]
    )
    training.append(t)
    epochs += e

# %%
# Get train history values.
train_history = dict()
for t in training:
    for key, value in t.history.items():
        if key in train_history:
            train_history[key].extend(value)
        else:
            train_history[key] = value.copy()

# %%
# Save training history.
with open(os.path.join(OUT_DIR, 'train_history.pkl'), 'wb') as f:
    pickle.dump(train_history, f)

# %%
# Save model.
model.save(os.path.join(OUT_DIR, 'model.h5'), include_optimizer=False)

# %%
# Plot loss evolution during training.
plt.plot(train_history['loss'], label='training')
plt.plot(train_history['val_loss'], label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('crossentropy loss')
plt.title('Loss during training')
plt.plot()

# %%
# Plot accuracy evolution during training.
plt.plot(train_history['accuracy'], label='training')
plt.plot(train_history['val_accuracy'], label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy during training')
plt.plot()

# %%
# Get predictions on test set.
preds = model.predict(ds_test)

# %%
# Compute CMC curve.
def compute_rank_accuracy(rank):
    acc = 0
    for p, (_, label) in zip(preds, ds_test.unbatch()):
        if label in np.argsort(p)[::-1][:rank]:
            acc += 1
    return acc / preds.shape[0]

cmc = [compute_rank_accuracy(r + 1) for r in range(N_LABELS)]

# %%
# Save CMC curve.
with open(os.path.join(OUT_DIR, 'cmc.pkl'), 'wb') as f:
    pickle.dump(cmc, f)

# %%
