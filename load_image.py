import numpy as np
from PIL import Image

def load_image(filepath: str) -> np.ndarray:
    im: Image.Image = Image.open(filepath)
    (width, height) = (im.width // 4, im.height // 4)
    im_resized = im.resize((width, height))
    return np.array(im_resized)
    