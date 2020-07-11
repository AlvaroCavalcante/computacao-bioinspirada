from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.exposure import rescale_intensity
# new_image = rescale_intensity(new_image, in_range=(0, 255))
import os

train_dataset = []
test_dataset = []

def generate_dataset(path, dataset, max_images, size=(64,64)):
    for filepath in os.listdir(path)[0:max_images]:
        loaded_image = image.load_img(path + filepath, target_size=size)
        
        img_array = np.asarray(loaded_image)
        img_array = img_array[:,:,0]
        
        reescaled_img_array = img_array/255
    
        dataset.append(reescaled_img_array)
        
    return dataset

train_dataset = generate_dataset('/home/alvaro/Documentos/dataset/training_set/cachorro/', train_dataset, 5)
train_dataset = generate_dataset('/home/alvaro/Documentos/dataset/training_set/gato/', train_dataset, 5)

test_dataset = generate_dataset('/home/alvaro/Documentos/dataset/test_set/cachorro/', test_dataset, 3)
test_dataset = generate_dataset('/home/alvaro/Documentos/dataset/test_set/gato/', test_dataset, 3)

kernel_sharpen = np.asmatrix([[0, -1, 0], [-1,5,-1], [0,-1,0]]) 
kernel_outline = np.asmatrix([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]])

def show_image(image):
    imgplot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

def get_img_with_padding(image):
    # pad formula = kernel_size - 1 / 2
    image = np.insert(image, 0, 0, axis=0)
    image = np.insert(image, 0, 0, axis=1)
    image = np.vstack([image, np.zeros(image.shape[1])])
    image = np.hstack([image, np.zeros((image.shape[0], 1))])
        
    return image

def convolution(image, kernel, stride, padding = False):
    initial_line = 0
    final_line = kernel.shape[0]
    new_image = []

    if padding:
        image = get_img_with_padding(image)
    
    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = kernel.shape[1]
        matrix_line = []

        while final_column <= image.shape[1]:
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                        
            kernel_result = np.dot(kernel, kernel_area)
            
            matrix_line.append(np.sum(kernel_result))
        
            initial_column += stride 
            final_column += stride
        
        new_image.append(matrix_line) 
        final_line += 1
        initial_line += 1

    return np.asmatrix(new_image)

def apply_relu(image):
    relu_img = image.copy() # deepcopy
    
    relu_img[relu_img < 0] = 0
    return relu_img

def max_pooling(image, stride, poll_size = 2):
    new_poll_image = []

    initial_line = 0
    final_line = poll_size
    
    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = poll_size
        matrix_line = []
        
        while final_column < (image.shape[1] - stride):
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                                    
            matrix_line.append(np.max(kernel_area))
            initial_column += stride
            final_column += stride
        
        new_poll_image.append(matrix_line)

        final_line += stride
        initial_line += stride  

    return np.asmatrix(new_poll_image)

def apply_conv(dataset, kernel):
    flatten_dataset = []
    for image in dataset:   
        conv_image = convolution(image, kernel, 1)
        show_image(conv_image * 255)
        
        conv_image_relu = apply_relu(conv_image)
        show_image(conv_image_relu * 255)
        
        conv_image = convolution(conv_image, kernel, 1)
        show_image(conv_image * 255)
        
        conv_image_relu = apply_relu(conv_image)
        show_image(conv_image_relu * 255)

        poll_image = max_pooling(conv_image_relu, 2, 2)
        show_image(poll_image * 255)
                    
        flatten_dataset.append(poll_image) # .flatten()

    return flatten_dataset

flatten_train = apply_conv(train_dataset, kernel_sharpen)
flatten_test = apply_conv(test_dataset, kernel_sharpen)

flatten_train = np.array(flatten_train) # [:,0,:]
flatten_test = np.array(flatten_test) # [:,0,:]

flatten_train = np.expand_dims(flatten_train, axis=-1)
flatten_test = np.expand_dims(flatten_test, axis=-1)

## training a neural network

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

data = [0] * (flatten_train.shape[0] // 2) + [1] * (flatten_train.shape[0] // 2) # 0 cachorro e 1 gato
train_label = pd.Series(data)

classificador = Sequential()

classificador.add(Dense(flatten_train.shape[0], input_shape=(63, 62, 1)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

classificador.fit(flatten_train, train_label, epochs = 25)

previsao = classificador.predict(flatten_test)

data = [0] * (flatten_test.shape[0] // 2) + [1] * (flatten_test.shape[0] // 2) # 0 cachorro e 1 gato
test_label = pd.Series(data)

previsao[previsao >= 0.5] = 1
previsao[previsao < 0.5] = 0

previsao = previsao[:, 0]

precisao = (test_label == previsao).sum() / len(test_label)

print('Precisão de teste', precisao)
