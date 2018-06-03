import numpy as np
import os
from scipy.misc import imread, imsave
import sys

def normalize(dir):
    sum = np.zeros((224, 224, 3))
    count = 0
    sum_std = np.zeros((224, 224, 3))
    # First pass to calculate mean
    for subdir in os.walk(dir):
        for img in os.listdir(subdir):
            if img.endswith(".jpg"):
                sum += imread(os.path.join(dir, img))
                count += 1
    mean = sum/count

    # Second pass to calculate std
    count = 0
    for subdir in os.walk(dir):
        for img in os.listdir(subdir):
            if img.endswith(".jpg"):
                sum_std += (imread(os.path.join(dir, img)) - mean)**2
                count += 1
    std = np.sqrt((sum_std - mean)/count)

    # Third pass to normalize images
    for subdir in os.walk(dir):
        for img in os.listdir(subdir):
            if img.endswith(".jpg"):
                imsave(img, (imread(os.path.join(dir, img)) - mean)/std)

    return mean, std


if __name__ == '__main__':
    directory = sys.argv[1]
    mean, std = normalize(directory)
    np.save(directory + '_mean.npy', mean)
    np.save(directory + '_std.npy', std)