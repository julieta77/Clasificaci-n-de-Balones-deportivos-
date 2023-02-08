import numpy as np 
import tensorflow as tf



longitud, altura = 100,100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = tf.keras.models.load_model(modelo)
cnn.load_weights(pesos_modelo) 

def predict(file):
    x = tf.keras.preprocessing.image.load_img(file, target_size=(longitud, altura))    #load_img(file, target_size=(longitud, altura))
    x =  tf.keras.preprocessing.image.img_to_array(x)  #img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer
    

print(predict('a.jpeg'))
