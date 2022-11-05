#%%

import glob
import os
import numpy as np
import tqdm
import cv2

#%%

def readfile(path, label):

    image_dir = sorted(glob.glob(os.path.join(path, "**/*.jpg"), recursive=True))
    x = np.zeros((len(image_dir), 3, 96, 96), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in tqdm.tqdm(enumerate(image_dir)):
        img = cv2.imread(file)
        x[i, :, :, :] = np.transpose(img, (2, 0, 1))
        if label:
            y[i] = int(image_dir[0].split("\\")[1])

    if label:
        return x, y
    else:
        return x
