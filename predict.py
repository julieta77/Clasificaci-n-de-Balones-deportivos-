import numpy as np 
import tensorflow as tf




longitud, altura = 150,150
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
    dicc = {'american_football': 0, 'baseball': 1, 'basketball': 2, 'billiard_ball': 3, 'bowling_ball': 4, 'cricket_ball': 5, 'football': 6, 'golf_ball': 7, 'hockey_ball': 8, 'hockey_puck': 9, 'rugby_ball': 10, 'shuttlecock': 11, 'table_tennis_ball': 12, 'tennis_ball': 13, 'volleyball': 14}
    for key, value in dicc.items():
        if value == answer:
            print(key)
    return answer
    


print(predict('tenis1.jpeg'))

