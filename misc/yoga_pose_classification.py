import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Rescaling
from keras.optimizers.legacy import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.src.layers import GlobalAveragePooling2D, BatchNormalization
from PIL import Image
from keras.src.optimizers.schedules import ExponentialDecay


def resize_images(train_dir, validation_dir, train_dir_resized, validation_dir_resized):
    resize_training(train_dir, train_dir_resized)
    resize_validation(validation_dir, validation_dir_resized)


def resize_training(train_dir, train_dir_resized):
    os.makedirs(train_dir_resized, exist_ok=True)

    for pose in os.listdir(train_dir):
        os.makedirs(train_dir_resized + '/' + pose, exist_ok=True)
        for img_file in os.listdir(train_dir + '/' + pose):
            try:
                img_path = os.path.join(train_dir + '/' + pose, img_file)
                img = Image.open(img_path)
                # img = img.resize((192, 192))
                img = img.convert('RGB')
                img.save(os.path.join(train_dir_resized + '/' + pose, img_file))
            except Exception as e:
                print(e)


def resize_validation(validation_dir, validation_dir_resized):
    os.makedirs(validation_dir_resized, exist_ok=True)

    for pose in os.listdir(validation_dir):
        os.makedirs(validation_dir_resized + '/' + pose, exist_ok=True)
        for img_file in os.listdir(validation_dir + '/' + pose):
            try:
                img_path = os.path.join(validation_dir + '/' + pose, img_file)
                img = Image.open(img_path)
                # img = img.resize((192, 192))
                img = img.convert('RGB')
                img.save(os.path.join(validation_dir_resized + '/' + pose, img_file))
            except Exception as e:
                print(e)


class Model:

    def __init__(self):
        self._model = None

    def build_model(self):
        model = Sequential()
        # can change to 500, 500 or whatever figsize we choose from run_movenet.py
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # output layer, 5 classes
        model.add(Dense(5, activation='softmax'))
        self._model = model
        self._model.build()

    def summary(self):
        self._model.summary()

    def compile(self):
        # lr_schedule = ExponentialDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=10000,
        #     decay_rate=0.9
        # )
        adam = Adam(learning_rate=0.001)
        # adam = Adam(learning_rate=lr_schedule)
        # adam = Adam(learning_rate=0.01)
        self._model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def generate_img_data(self):
        # set batch size
        bs = 30

        train_dir = './train'

        validation_dir = './test'

        datagen = ImageDataGenerator(
            rescale=1./255,
            dtype='float32',
            # rotation_range=30,  # try with 10
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # fill_mode='nearest'
        )
        # validation_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     dtype='float32'
        # )

        train_generator = datagen.flow_from_directory(
            # train_dir,
            train_dir,
            batch_size=bs,
            class_mode='categorical',
            target_size=(300, 300),
            classes=['downdog', 'goddess', 'plank', 'tree', 'warrior2'],
            # subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            # validation_dir,
            validation_dir,
            batch_size=bs,
            class_mode='categorical',
            target_size=(300, 300),
            classes=['downdog', 'goddess', 'plank', 'tree', 'warrior2'],
            # subset='validation'
        )

        return train_generator, validation_generator

    def train(self, train_generator, validation_generator, bs=30):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        # early_stopping = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3)
        model_checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True)

        history = self._model.fit(
            train_generator,
            validation_data=validation_generator,
            # steps_per_epoch=len(train_generator),  # number of batches per epoch during training
            epochs=25,  # try to increase this and see how it changes
            # epoch 13 = 0.9241 accuracy
            # epoch 17 = 0.9363 accuracy
            # validation_steps=len(validation_generator),  # number of batches per epoch during validation
            verbose=1,
            batch_size=bs,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )

        return history

    def save(self, filepath):
        self._model.save(filepath)

    # def evaluate(self):
    #   return self._model.evaluate()


####### SAVE MODEL #######
cnn_model = Model()
cnn_model.build_model()
cnn_model.summary()
cnn_model.compile()
train_generator, validation_generator = cnn_model.generate_img_data()
history = cnn_model.train(train_generator, validation_generator)

cnn_model.save('stick_cnn_25epochs_moreComplex_callbacks_batchNormalization.h5')
# cnn_model.save('stick_yoga_poses_cnn_data_augmentation.h5')

### PREDICTION ###
# loaded_model = load_model('stick_cnn_15epochs_less_complex_callbacks.h5')
#
# img_path = './final_test/stick/3.jpg'
# img = image.load_img(img_path, target_size=(300, 300))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0
#
# predictions = loaded_model.predict(img_array)
# np.set_printoptions(suppress=True, precision=4)
# print(predictions)
# label = np.argmax(predictions)
#
# entries = os.listdir('./sticks')
# labels = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}
# # labels = {i: label for i, label in enumerate(entries) if label != '.DS_Store'}
# # labels = {i - 1 if i > 1 else i: label for i, label in labels.items()}
# print(labels)
# print(labels[label])
