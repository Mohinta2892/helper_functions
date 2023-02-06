import tifffile
from tifffile import imsave, imwrite
import numpy as np
import matplotlib.pyplot as plt

# read .npy
a = np.load('/Users/sam/Downloads/instance_seg_DanSamiaTest.npy')

# write tiff, imsave is deprecated
imwrite('/Users/sam/Downloads/instance_seg_DanSamiaTest.tiff', a)

# check if you can it back
t = tifffile.imread('/Users/sam/Downloads/instance_seg_DanSamiaTest.tiff')
# plot a slice from 3D array

plt.imshow(t[1, ...], cmap='jet')
