import sys
import os
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K 
import tensorflow as tf
from PIL import Image
import scipy


K.clear_session() 
 
train_rut = './train' 
test_rut = './test'


## Parametros

epocas = 10 
altura, longitud  = 150 , 150
batch_size = 32
pasos = 2000
pasos_validacion = 210 
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool= (2,2)
clases = 15 
lr = 0.0004

##p

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip =True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)

imagen_train = train_datagen.flow_from_directory(
    train_rut,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode= 'categorical'
)


imagen_test = test_datagen.flow_from_directory(
    test_rut,
    target_size=(altura,longitud),
    batch_size= batch_size,
    class_mode='categorical'
)


## red neuronal 

modelo = Sequential()

#capa 1 

modelo.add(Convolution2D(filtrosConv1,tamano_filtro1,input_shape=(longitud,altura,3),activation='relu'))

modelo.add(MaxPooling2D(pool_size=tamano_pool))

modelo.add(Convolution2D(filtrosConv2,tamano_filtro2))

modelo.add(MaxPooling2D(pool_size=tamano_pool))

modelo.add(Flatten())

modelo.add(Dense(180,activation='relu'))
modelo.add(Dense(84,activation='relu'))
modelo.add(Dropout(0.4))
modelo.add(Dense(clases,activation='softmax'))

modelo.compile(loss='categorical_crossentropy',
            optimizer=optimizers.adam_v2.Adam(learning_rate=lr),
            metrics=['accuracy'])


modelo.fit(imagen_train,
        steps_per_epoch=pasos,
        epochs=epocas,
        validation_data=imagen_test,
        validation_freq=pasos_validacion) 


print(imagen_train.class_indices)

modelo_rut = './modelo/'
if not os.path.exists(modelo_rut):
  os.mkdir(modelo_rut)


modelo.save('./modelo/modelo.h5')
modelo.save_weights('./modelo/pesos.h5')

