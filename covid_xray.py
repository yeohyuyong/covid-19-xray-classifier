from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 input_shape=(128,128,3),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,
                activation='relu',
                input_shape=(128,128)))
          
model.add(Dense(units=128,
                activation='relu'))

model.add(Dense(units=1,
                activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('xray_dataset_covid19/train',
                                                 target_size=(128,128),
                                                 batch_size=16,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('xray_dataset_covid19/test',
                                            target_size=(128, 128),
                                            batch_size=16,
                                            class_mode='binary')


model.fit_generator(training_set,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=800)



