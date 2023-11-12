import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Rescaling
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from PIL import Image

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
        img = img.resize((192, 192))
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
        img = img.resize((192, 192))
        img = img.convert('RGB')
        img.save(os.path.join(validation_dir_resized + '/' + pose, img_file))
      except Exception as e:
        print(e)
  

class Model():
  
  def __init__(self):
    self._model = None
    
  def build_model(self):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(192,192,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # output layer, 5 classes
    model.add(Dense(5, activation='softmax'))
    self._model = model
    
  def summary(self):
    self._model.summary()
    
  def compile(self):
    adam = Adam(learning_rate=0.001)
    self._model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
  def generate_img_data(self):
    # set batch size
    bs = 30
    
    train_dir = './DATASET2/TRAIN'
    train_dir_resized = './DATASET2/TRAIN_RESIZED'
    
    validation_dir = './DATASET2/TEST'
    validation_dir_resized = './DATASET2/TEST_RESIZED'
    
    # uncomment line below if you haven't run before
    # resize_images(train_dir, validation_dir, train_dir_resized, validation_dir_resized)
    
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      dtype='float32'
    )
    validation_datagen = ImageDataGenerator(
      rescale=1./255,
      dtype='float32'
    )
    
    train_generator = train_datagen.flow_from_directory(
      # train_dir,
      train_dir_resized,
      batch_size=bs,
      class_mode='categorical',
      target_size=(192, 192),
      subset='training'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
      # validation_dir,
      validation_dir_resized,
      batch_size=bs,
      class_mode='categorical',
      target_size=(192, 192),
      # subset='validation'
    )
    
    return train_generator, validation_generator
  
  def train(self, train_generator, validation_generator, bs=30):
    history = self._model.fit(
      train_generator,
      validation_data=validation_generator,
      steps_per_epoch=len(train_generator),
      epochs=10,
      validation_steps=len(validation_generator),
      # verbose=2
      )
    
    return history
  
  def save(self, filepath):
    self._model.save(filepath)
  
  # def evaluate(self):
  #   return self._model.evaluate()
  

model = Model()
model.build_model()
model.summary()
model.compile()
train_generator, validation_generator = model.generate_img_data()
history = model.train(train_generator, validation_generator)

model.save('yoga_poses_cnn.h5')
model.save('yoga_poses_cnn.keras')


### PREDICTION ###
# loaded_model = load_model('yoga_poses_cnn.h5')

# img_path = './goddess_test_2.jpg'
# img = image.load_img(img_path, target_size=(192, 192))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# predictions = loaded_model.predict(img_array)
# np.set_printoptions(suppress=True, precision=4)
# print(predictions)
# label = np.argmax(predictions)

# entries = os.listdir('./DATASET/TRAIN')
# labels = {i:label for i, label in enumerate(entries)}
# print(labels)
# print(labels[label])
    

    
