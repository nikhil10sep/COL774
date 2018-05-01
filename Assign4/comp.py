from os import listdir
from os.path import isfile, join
import numpy as np


def save_pred(pred, filename, classes):
    file = open(filename, "w")
    file.write("ID,CATEGORY\n")
    
    for i in range(len(pred)):
        file.write(str(i) + ',' + classes[int(pred[i])] + '\n')
    
    file.close()
    return


def read_train_data(files):
    x = (np.load('train/' + files[0]))
    whole_data = np.empty((x.shape[0], x.shape[1] + 1))
    whole_data[:, :-1] = x;
    whole_data[:, -1] = 0;

    for i in range (1, len(files)):
        x = np.load('train/' + files[i])
        data = np.empty((x.shape[0], x.shape[1] + 1))
        data[:, :-1] = x;
        data[:, -1] = i;
        whole_data = np.append(whole_data, data, axis=0);
        
    return whole_data


def read_test_data():
    return np.load('test/test.npy')


files = [f for f in listdir('train') if isfile(join('train', f))]


whole_data = read_train_data(files)
np.random.shuffle(whole_data);
X = whole_data[:, :-1] / 255
y = whole_data[:, -1].astype(int)

X_test = read_test_data() / 255

classes = [s.split('.')[0] for s in files]


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.constraints import max_norm
from keras.layers import BatchNormalization
from keras.models import load_model

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(y)

X_s = X.reshape(X.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_initializer = 'glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='softmax', kernel_initializer = 'glorot_uniform'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())



from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(horizontal_flip=True)

datagen_train.fit(X_s)

model.fit_generator(datagen_train.flow(X_s, dummy_y, batch_size=100), epochs=12, verbose=0)

model.save('best.h5')

score = model.evaluate(X_s, dummy_y);
print('Train accuracy = ', score[1] * 100)

y_pred = model.predict(X_test.reshape(X_test.shape[0], 28, 28, 1))
y_pred = np.argmax(y_pred, axis=1)

save_pred(y_pred, 'best.txt', classes)

