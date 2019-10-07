import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:5000]
test_labels = test_labels[:5000]

train_images = train_images[:5000].reshape(-1, 28*28) / 255.0
test_images = test_images[:5000].reshape(-1, 28 * 28) / 255.0

# simple model
model = tf.keras.models.Sequential([
	keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(10, activation=tf.keras.activations.softmax)
	])

model.compile(optimizer="adam", loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

check_point_path = "/media/Application_and_Program/DeepLearningProject/advance_applied_deeplearning/save_load_models" \
                   "/checkpoint/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(check_point_path, save_weights_only=True, verbose=1)

model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])


### model2 untrained
model2 = tf.keras.models.Sequential([
	keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model2.compile(optimizer='adam',
               loss=tf.keras.losses.sparse_categorical_crossentropy,
               metrics=['accuracy'])

loss, acc = model2.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

## load weights from the "model"
model2.load_weights(check_point_path)
loss, acc = model2.evaluate(test_images, test_labels)
print(f"[INFO] Second model, accuracy: {100*acc}")

