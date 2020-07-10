# Notes:
# - rosbag images are RGB png's (no alpha layer)

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from preprocessing import preprocess_zhang

def load_image(filepath: str) -> np.ndarray:
    '''
    Use PIL to load and resize a rosbag image.

    Parameters:
        filepath (str): Path to file from current working directory

    Returns:
        np.ndarray: Should be a (512, 640, 3) numpy array
    '''
    im: Image.Image = Image.open(filepath)
    (width, height) = (im.width // 4, im.height // 4)
    im_resized = im.resize((width, height))
    return np.array(im_resized)

# %% 
im = preprocess_zhang(load_image('img1.png'), normalize=True)
plt.imshow(im)

# %%
