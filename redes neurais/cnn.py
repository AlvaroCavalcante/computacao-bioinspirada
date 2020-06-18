from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

image = Image.open("/home/alvaro/Documentos/mestrado/computação bio/redes neurais/img_dataset/dog_small.png")

image = np.asarray(image)
image = image[:,:,0]

# imgplot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.show()

kernel = np.asmatrix([[0, -1, 0], [-1,5,-1], [0,-1,0]]) # sharpen
# kernel = np.asmatrix([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]]) # top sobel

position = 3
stride = 1
padding = 1

new_image = np.zeros(shape=(399, 300))

linha_inicial = 0
linha_final = 3

while position < image.shape[0]:    
    coluna_inicial = 0
    coluna_final = 3
    
    for i in range(image.shape[1]):
        kernel_area = image[linha_inicial:linha_final, coluna_inicial:coluna_final]
        
        if kernel_area.shape != kernel.shape:
            kernel_area = np.vstack([kernel_area, np.asmatrix([[0,0,0]])]) if kernel_area.shape[0] != kernel.shape[0] else kernel_area  
            kernel_area = np.hstack([kernel_area, np.asmatrix([[0],[0],[0]])]) if kernel_area.shape[1] != kernel.shape[1] else kernel_area  

        kernel_result = np.dot(kernel, kernel_area)
        
        new_image[linha_inicial, coluna_inicial] = np.sum(kernel_result)
    
        coluna_inicial += padding
        coluna_final += padding
    
    linha_final += stride
    linha_inicial += stride    

# new_image = rescale_intensity(new_image, in_range=(0, 255))

imgplot = plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
plt.show()

# img = Image.fromarray(new_image)
