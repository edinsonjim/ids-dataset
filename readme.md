# Identification Cards Dataset

This repository contains images downloaded from the internet to create a small dataset and apply deep learning to classify images.

## Structure

```bash
.
├── readme.md
├── test
│   ├── 0001.jpeg
│   ├── 0002.jpg
│   ├── 0003.jpg
│   ├── 0004.jpg
│   └── 0005.png
└── training
    ├── italy
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   ├── 0003.jpg
    │   ├── 0004.jpg
    │   ├── 0005.jpg
    │   ├── 0006.jpg
    │   ├── 0007.jpg
    │   └── 0008.jpg
    ├── peru
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   ├── 0003.jpg
    │   ├── 0004.jpg
    │   ├── 0005.jpg
    │   ├── 0006.jpg
    │   ├── 0007.jpg
    │   ├── 0008.jpg
    │   ├── 0009.jpg
    │   ├── 0010.jpg
    │   └── 0011.jpg
    └── spain
        ├── 0001.jpg
        ├── 0002.jpg
        ├── 0003.jpg
        ├── 0004.jpg
        ├── 0005.jpg
        └── 0006.jpg
```

## Use with keras

You can use `image_dataset_from_directory` utility to generate the dataset.

```python
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    WORK_DIR,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    WORK_DIR,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
```
