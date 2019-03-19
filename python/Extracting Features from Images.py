#!/usr/bin/env python
# coding: utf-8

# ## Extracting features from images 
# ### 1. Extracting features from pixel intensities
# First look at the pixel intensity for a pixel image of the number 0

# In[4]:


from sklearn import datasets
digits = datasets.load_digits()

print (digits.target[0])
print(digits.images[0])
print('Feature vector:\n', digits.images[0].reshape(-1, 64))


# ####  As we can see from the result, the vector space is not sparse, and for a tiny image a vector with size length*width is in need. 
# So modern computer vision applications frequently use 
#     1. hand-engineered feature extraction methods
#         - applicable to many different problems
#     2. unsupervised learning

# ### 2. Extracting points of interest as features
# - extract important points like edges and corners

# In[25]:


import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist


# In[69]:


def show(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, "or")
    plt.xlim(1, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()


# In[70]:


from skimage import io
lain = io.imread('pic/lain.jpg')
lain = equalize_hist(rgb2gray(lain))
corners = corner_peaks(corner_harris(lain), min_distance=2)
show(corners, lain)


# #### Plot edges

# In[72]:


from skimage.filters import roberts, scharr, prewitt
edge_roberts = roberts(lain)
fig, ax = plt.subplots()
fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
ax.imshow(edge_roberts, cmap=plt.cm.gray)
ax.axis("off")
plt.show()


# #### Other edge methods documented in skimage

# In[75]:


x, y = np.ogrid[:100, :100]
# Rotation-invariant image with different spatial frequencies

edge_sobel = sobel(lain)
edge_scharr = scharr(lain)
edge_prewitt = prewitt(lain)

diff_scharr_prewitt = edge_scharr - edge_prewitt
diff_scharr_sobel = edge_scharr - edge_sobel
max_diff = np.max(np.maximum(diff_scharr_prewitt, diff_scharr_sobel))

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(lain, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_scharr, cmap=plt.cm.gray)
ax[1].set_title('Scharr Edge Detection')

ax[2].imshow(diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
ax[2].set_title('Scharr - Prewitt')

ax[3].imshow(diff_scharr_sobel, cmap=plt.cm.gray, vmax=max_diff)
ax[3].set_title('Scharr - Sobel')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


# ### 3. SIFT and SURF
# #### Scale-Invariant Feature Transform
# - for extracting features from images which are not that sensitive to the scale, rotation, and illumination of the image
# - Each SIFT feature, or descriptor, is a vector that describes edges and corners in a region of an image
# - SIFT also captures information about the composition of each point of interest and its surroundings
# 
# #### Speeded-Up Robust Features
# - can be
# computed more quickly than SIFT
# - more effective at recognizing features
# across images that have been transformed in certain ways.
# 
# #### Will explore more in Clusternig with k-Means

# In[79]:


import mahotas as mh
from mahotas.features import surf
image = mh.imread('pic/lain.jpg', as_grey=True)
print ('The first SURF descriptor:\n', surf.surf(image)[0])
print ('Extracted %s SURF descriptors' % len(surf.surf(image)))

