#import numpy as np
import matplotlib.pyplot as plt
#from scipy import ndimage as ndi
from PIL import Image
import numpy
from skimage import feature
import cv2


# Generate noisy image of a square
#im = np.zeros((128, 128))
#im[32:-32, 32:-32] = 1

#im = ndi.rotate(im, 15, mode='constant')
#im = ndi.gaussian_filter(im, 4)
#im += 0.2 * np.random.random(im.shape)

# import the file
PATH="C://Users//chandar_s//Pictures//TestSheet2.tif"
im = numpy.asarray(Image.open(PATH).convert('L'))

#print (im_rbg.shape)

#from skimage.color import rgb2gray
#im = rgb2gray(im_rbg)

#print (im.shape)

# Compute the Canny filter for two values of sigma
#edges1 = feature.canny(im)
#edges2 = feature.canny(im, sigma=3)

edges1 = cv2.Canny(im, 100, 255) 
edges2 = cv2.Canny(im) 

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Grayscale image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()

quit()

