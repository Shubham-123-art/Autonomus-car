import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle
import pandas as pd
import random
import cv2

with open(r"C:\Users\pc\Desktop\finding-lanes\train.p", 'rb') as f:
    train_data = pickle.load(f)
with open(r"C:\Users\pc\Desktop\finding-lanes\valid.p", 'rb') as f:
    val_data = pickle.load(f)
with open(r"C:\Users\pc\Desktop\finding-lanes\test.p", 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
X_val, y_val = val_data['features'], val_data['labels']

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

assert(X_train.shape[0] == y_train.shape[0]
       ), "The number of images is not equal to the number of labels"
assert(X_val.shape[0] == y_val.shape[0]
       ), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]
       ), "The number of images is not equal to the number of labels"


assert(X_train.shape[1:] == (
    32, 32, 3)), "The dimensions of image is not equal to the number of labels"

assert(X_val.shape[1:] == (
    32, 32, 3)), "The dimensions of image is not equal to the number of labels"

assert(X_test.shape[1:] == (
    32, 32, 3)), "The dimensions of image is not equal to the number of labels"


data = pd.read_csv(r"C:\Users\pc\Desktop\finding-lanes\signnames.csv")
num_of_samples = []
cols = 5
num_classes = 43
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(
            0, (len(x_selected)-1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j)+'_'+row['SignName'])
            num_of_samples.append(len(x_selected))


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of train dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of images")
plt.show()

plt.imshow(X_train[1000])
plt.axis('off')
print(X_train[1000].shape)
print(y_train[1000])


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_val = np.array(list(map(preprocessing, X_val)))
plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
print(X_train.shape)

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)


def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model


model = leNet_model()
print(model.summary())

model.fit(X_train, y_train, epochs=10, validation_data=(
    X_val, y_val), batch_size=400, verbose=1, shuffle=1)
