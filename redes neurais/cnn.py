from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

image = Image.open("/home/alvaro/Documentos/mestrado/computação bio/redes neurais/img_dataset/gray_dog.png")

image = np.asarray(image)
image = image[:,:,0]

imgplot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()

# kernel = np.asmatrix([[0, -1, 0], [-1,5,-1], [0,-1,0]]) #sharpen
# kernel = np.asmatrix([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]]) # top sobel
kernel = np.asmatrix([[2, 2, 4, 2, 2], [1,1,2,1,1], [0,0,0,0,0], [-1,-1,-2,-1,-1], [-2,-2,-4,-2,-2]]) # top sobel

x = np.expand_dims(kernel, axis=0)

position = 3
stride = 1
padding = 1

new_image = np.zeros(shape=(1332, 999))

image = image[0:1332, 0:999]

linha_inicial = 0
linha_final = 3

while position < image.shape[0]:    
    coluna_inicial = 0
    coluna_final = 3
    
    for i in range(image.shape[1] - 3):
        kernel_area = image[linha_inicial:linha_final, coluna_inicial:coluna_final]
        kernel_result = np.dot(x, kernel_area)
        
        new_image[linha_inicial, coluna_inicial] = np.sum(kernel_result[0,:,:])
    
        coluna_inicial += padding
        coluna_final += padding
    
    linha_final += stride
    linha_inicial += stride    

new_image = rescale_intensity(new_image, in_range=(0, 255))

imgplot = plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
plt.show()