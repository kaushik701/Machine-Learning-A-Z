#%%
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# %%
train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range = 0.2,horizontal_flip=True)
train_set = train_datagen.flow_from_directory('dataset/train_set',target_size=(64,64),batch_size=32,class_mode='binary')
# %%
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_set = test_datagen.flow_from_directory('',target_size=(64,64),batch_size=32,class_mode='binary')
#%%
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size = 2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(x = train_set,validation_data = test_set,epochs=25)
#%%
test_image = image.load_img('',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)