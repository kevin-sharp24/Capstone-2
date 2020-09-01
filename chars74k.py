#%% import libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage import color
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

#%% image preprocessing - standard
SAMPLES = 7700
ROWS = 64
COLS = 64

images = np.zeros(shape=(SAMPLES, ROWS, COLS, 3))
labels = np.zeros(shape=(SAMPLES,), dtype=int)
img_num = 0
for img_sample_dir in sorted(glob.glob('English\Img\GoodImg\Bmp\*')):
    for image in glob.glob(img_sample_dir + '\*'):
        images[img_num] = resize(imread(image), (COLS, ROWS, 3))
        
        '''
        grab last two characters of sample directory, which will be a pair of numbers (e.g. 01),
        convert that numeric string to an integer, then subtract 1 and set that value to be the label
        '''
        labels[img_num] = int(img_sample_dir[-2:]) - 1
        img_num += 1

#%% train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.75)

#%% layers
model = Sequential()
model.add(Conv2D(128, kernel_size=3, padding='same', input_shape=(ROWS, COLS, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(62, activation='softmax'))

#%% run model
optimizer = SGD(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))
model.summary()

#%% Show image
plt.imshow(images[2])
plt.show()

#%% Get image dimensions (shown in variable explorer)

img = resize(color.rgb2gray(io.imread('English\Img\GoodImg\Bmp\Sample001\img001-00003.png')), (64, 64))
plt.imshow(img)
plt.show()