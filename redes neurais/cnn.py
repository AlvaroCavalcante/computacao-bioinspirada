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
# kernel = np.asmatrix([[-1, -1, -1], [-1,8,-1], [-1,-1,-1]]) # outline

stride = 1
padding = 1

new_image = np.zeros(shape=(399, 300))

initial_line = 0
final_line = 3

while final_line <= image.shape[0]:    
    initial_column = 0
    final_column = 3
    
    for i in range(image.shape[1]):
        kernel_area = image[initial_line:final_line, initial_column:final_column]
                    
        if kernel_area.shape != kernel.shape:
            kernel_area = np.vstack([kernel_area, np.asmatrix([[0,0,0]])]) if kernel_area.shape[0] != kernel.shape[0] else kernel_area  
            kernel_area = np.hstack([kernel_area, np.asmatrix([[0],[0],[0]])]) if kernel_area.shape[1] != kernel.shape[1] else kernel_area  

        kernel_result = np.dot(kernel, kernel_area)
        
        new_image[initial_line, initial_column] = np.sum(kernel_result)
    
        initial_column += padding
        final_column += padding
    
    final_line += stride
    initial_line += stride  

# new_image = rescale_intensity(new_image, in_range=(0, 255))
position = 2
stride = 2
padding = 2

imgplot = plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
plt.show()

new_poll_image = np.zeros(shape=(399, 300))

initial_line = 0
final_line = 2

while position <= image.shape[0]:    
    try:
        initial_column = 0
        final_column = 2
        
        for i in range(image.shape[1]):
            kernel_area = image[initial_line:final_line, initial_column:final_column]
                                    
            new_poll_image[initial_line, initial_column - 1] = np.max(kernel_area)
        
            initial_column += padding
            final_column += padding
        
        final_line += stride
        initial_line += stride  
    except:
        break
