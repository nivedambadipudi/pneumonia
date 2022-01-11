import os
import time
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
'''
train_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/train'
test_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/test'
valid_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/val'
'''
CATEGORIES = ["normal", "pneumonia"]
'''
img_height = 500
img_width = 500
epochs = 25
batch_size = 16

for category in CATEGORIES:
	path = os.path.join(train_path, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
		plt.imshow(img_array, cmap='gray')
		plt.show()
		break
	break

image_gen = ImageDataGenerator(rescale=1./255,
							shear_range=0.2,
							zoom_range=0.2,
							horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1/255)

train = image_gen.flow_from_directory(
		train_path,
		target_size=(img_height, img_width),
		color_mode='grayscale',
		batch_size=batch_size,
		class_mode='binary',
		)

test = image_gen.flow_from_directory(
		test_path,
		target_size=(img_height, img_width),
		shuffle=False,
		color_mode='grayscale',
		class_mode='binary',
		batch_size=32
		)

valid = image_gen.flow_from_directory(
		valid_path,
		target_size=(img_height, img_width),
		color_mode='grayscale',
		class_mode='binary',
		batch_size=8
		)

dense_layers = [2]
layer_sizes = [120]
conv_layers = [3]

for dense_layer in dense_layers:
	for conv_layer in conv_layers:
		for layer_size in layer_sizes:

			model = Sequential()

			model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1), padding='same'))
			model.add(MaxPooling2D(pool_size=(2, 2)))

			for l in range(1, conv_layer):
				model.add(Conv2D(32*(2**l), (3, 3), activation='relu', padding='same'))
				model.add(MaxPooling2D(pool_size=(1+l, 1+l)))

			model.add(Flatten())

			for l in range(1, dense_layer+1):
				model.add(Dense(activation='relu', units=layer_size/l))

			model.add(Dropout(rate=0.2))
			model.add(Dense(activation='sigmoid', units=1))

			model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
			model.summary()

			weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
			cw = dict(zip(np.unique(train.classes), weights))

			early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

			learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
				patience=1, verbose=1, factor=0.5, min_lr=0.000001)

			callbacks_list = [early, learning_rate_reduction]

			history = model.fit(train, epochs=epochs, validation_data=valid)

			fp = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/model.h5'
			model.save(fp)
'''

model = load_model('model.h5')
'''
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(range(len(accuracy)), accuracy, color='blue', label='Training accuracy')
plt.plot(range(len(accuracy)), val_accuracy, color='red', label='Validation accuracy')
plt.xlabel('Epoch No.')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(accuracy)), loss, color='blue', label='Training loss')
plt.plot(range(len(accuracy)), val_loss, color='red', label='Validation loss')
plt.xlabel('Epoch No.')
plt.ylabel('loss')
plt.legend()
plt.show()

test_accuracy = model.evaluate(test)
print('The testing accuracy is: ', test_accuracy[1]*100, '%')

true = test.classes
preds = model.predict(test, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

cm = confusion_matrix(true, np.round(predictions))
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("True", color="royalblue", fontsize=35, fontweight=700)
ax.set_xlabel("Prediction", color="royalblue", fontsize=35, fontweight=700)
plt.yticks(rotation=0)
plt.show()

print(classification_report(y_true=test.classes, y_pred=predictions, target_names=['NORMAL', 'PNEUMONIA']))
'''

def prepare(filepath):
	IMG_SIZE = 500
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict([prepare('normal.jpg')])
print(CATEGORIES[int(prediction[0][0])])