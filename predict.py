#############################################################################  Importando librerias ####################################################################################################################

import numpy as np 
import tensorflow as tf



longitud, altura = 150,150 #Tiene que ser la misma altura y longitud que definimos en entrenar.py

modelo_rut = './modelo/modelo.h5' 
pesos_modelo = './modelo/pesos.h5'

modelo = tf.keras.models.load_model(modelo_rut) # Cargando el modelo
modelo.load_weights(pesos_modelo) 


def predict(file):
    """ Esta función el parámetro a pasar va a ser una imagen. Devuelve una predicción correspondiente a la imagen """

    img = tf.keras.preprocessing.image.load_img(file, target_size=(longitud, altura))   
    img =  tf.keras.preprocessing.image.img_to_array(img)  
    img = np.expand_dims(img, axis=0)
    array = modelo.predict(img)
    result = array[0]
    answer = np.argmax(result)
    dicc = {'american_football': 0, 'baseball': 1, 'basketball': 2, 'billiard_ball': 3, 'bowling_ball': 4, 'cricket_ball': 5, 'football': 6, 'golf_ball': 7, 'hockey_ball': 8, 'hockey_puck': 9, 'rugby_ball': 10, 'table_tennis_ball': 11, 'tennis_ball': 12, 'volleyball': 13}
    for key, value in dicc.items():
        if value == answer:
            print(key)
    return answer 
    

#print(predict('volleyball.jpeg'))