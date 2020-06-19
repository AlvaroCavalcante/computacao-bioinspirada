from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
# from skimage.exposure import rescale_intensity
# new_image = rescale_intensity(new_image, in_range=(0, 255))
import os

train_dataset = []
test_dataset = []

def generate_dataset(path, dataset, size=(128,128)):
    for filepath in os.listdir(path):
        loaded_image = image.load_img(path + filepath, target_size=size)
        
        img_array = np.asarray(loaded_image)
        img_array = img_array[:,:,0]
        
        reescaled_img_array = img_array/255
    
        dataset.append(reescaled_img_array)
        
    return dataset

train_dataset = generate_dataset('/home/alvaro/Documentos/dataset/example_set/train/cachorro/', train_dataset)
train_dataset = generate_dataset('/home/alvaro/Documentos/dataset/example_set/train/gato/', train_dataset)

test_dataset = generate_dataset('/home/alvaro/Documentos/dataset/example_set/test/cachorro/', test_dataset)
test_dataset = generate_dataset('/home/alvaro/Documentos/dataset/example_set/test/gato/', test_dataset)

kernel_sharpen = np.asmatrix([[0, -1, 0], [-1,5,-1], [0,-1,0]]) 
kernel_outline = np.asmatrix([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]])

def show_image(image):
    imgplot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

def convolution(image, kernel, stride, padding):
    initial_line = 0
    final_line = kernel.shape[0]
    new_image = []

    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = 3
        matrix_line = []

        for i in range(image.shape[1] // stride):
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                        
            if kernel_area.shape != kernel.shape:
                kernel_area = np.vstack([kernel_area, np.asmatrix([[0,0,0]])]) if kernel_area.shape[0] != kernel.shape[0] else kernel_area  
                kernel_area = np.hstack([kernel_area, np.asmatrix([[0],[0],[0]])]) if kernel_area.shape[1] != kernel.shape[1] else kernel_area  

            kernel_result = np.dot(kernel, kernel_area)
            
            matrix_line.append(np.sum(kernel_result))
        
            initial_column += stride 
            final_column += stride
        
        new_image.append(matrix_line) 
        final_line += padding
        initial_line += padding  

    return np.asmatrix(new_image)

def max_pooling(image, stride, padding):
    new_poll_image = []

    initial_line = 0
    final_line = 2
    
    while final_line <= image.shape[0]:    
        initial_column = 0
        final_column = 2
        matrix_line = []
        
        for i in range((image.shape[1] // stride) - stride):
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                                    
            matrix_line.append(np.max(kernel_area))
            initial_column += stride
            final_column += stride
        
        new_poll_image.append(matrix_line)

        final_line += padding
        initial_line += padding  

    return np.asmatrix(new_poll_image)

def apply_conv(dataset, kernel):
    flatten_dataset = []
    for image in dataset:   
        conv_image = convolution(image, kernel, 1, 1)
        # show_image(conv_image)
        
        poll_image = max_pooling(conv_image, 2, 2)
        # show_image(poll_image)
            
        flatten_dataset.append(poll_image.flatten())

    return flatten_dataset

flatten_train = apply_conv(train_dataset, kernel_sharpen)
flatten_test = apply_conv(test_dataset, kernel_sharpen)

flatten_train = np.array(flatten_train)[:,0,:]
flatten_test = np.array(flatten_test)[:,0,:]
