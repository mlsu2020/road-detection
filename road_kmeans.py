# Road Detection
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_closing, binary_opening
from skimage import measure

from load_image import load_image
from diamond import diamond

# File operations
import glob
import os
import copy
from tqdm import tqdm
from PIL import Image

def cluster_image(im: np.ndarray, K = 3) -> np.ndarray:
    X = im.reshape(-1, 3)
    batch_size = X.shape[0] // 5
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=batch_size).fit(X)

    # Generate clustered image
    im_c = np.zeros(im.shape)
    im_c = im_c.reshape(-1, 3)
    for label in np.unique(kmeans.labels_):
        im_c[kmeans.labels_ == label] = kmeans.cluster_centers_[label]
    im_c = im_c.reshape(im.shape) / 255
    
    return im_c, kmeans

def get_largest_component(cc: np.ndarray) -> np.ndarray:
    cc_flat = cc.reshape(-1)
    unique_labels = np.sort(np.unique(cc))
    counts = []
    for label in unique_labels:
        counts.append(len(cc_flat[cc_flat==label]))
    ind_max = np.argmax(counts[1:]) # Ignore 0. That's background
    return unique_labels[1:][ind_max]

def get_road_mask(im: np.ndarray) -> np.ndarray:
    # Gaussian blur to make the road surface look more even
    im_blur = gaussian_filter(im, sigma=1)

    # K-Means clustering to get road
    im_c, kmeans = cluster_image(im_blur)
    mean_labels = np.mean(kmeans.cluster_centers_, axis=1) / 255
    midtone_label = np.argsort(mean_labels)[1]
    midtone_value = kmeans.cluster_centers_[midtone_label] / 255

    # midtone mask (assume midtone contains road)
    mask = (im_c == midtone_value)
    mask = mask.astype('float64').mean(axis=2)

    # Binary closing - close noise
    kernel = diamond(3)
    mask_closed = binary_closing(mask, kernel)

    # Binary opening - disconnect thinly connected parts
    mask_closed = binary_opening(mask, kernel)

    # Connected components
    cc = measure.label(mask_closed, background=0)

    # Assume the largest group is the road
    largest_cc = get_largest_component(cc)
    mask_main_cc = np.zeros(cc.shape)
    mask_main_cc[cc == largest_cc] = 1

    # Get connected components of the reverse mask to fill in holes
    reverse_mask = 1 - mask_main_cc
    cc_rev = measure.label(reverse_mask, background=0)
    cc_rev_flat = cc_rev.reshape(-1)
    unique_labels = np.unique(cc_rev)
    counts = []
    for label in unique_labels:
        counts.append(len(cc_rev_flat[cc_rev_flat==label]))
    max_size = np.max(counts)
    for label in unique_labels:
        if counts[label] < max_size:
            cc_rev[cc_rev==label] = 0

    cc_rev[cc_rev > 0] = 1
    final_mask = 1 - cc_rev
    final_mask = final_mask.astype(np.float)
    return final_mask

if __name__ == '__main__':
    # image_names = glob.glob('all_images/img*.png')
    num_images = len(glob.glob('all_images/img*.png'))
    image_names = ['all_images/img{num}.png'.format(num = n) for n in range(1, num_images + 1)]
    for image in tqdm(image_names):
        file_name = os.path.basename(image)[:-4]
        im = load_image(image)

        # Run road detection algorithm
        road_mask = get_road_mask(im)

        # Get overlayed image
        overlayed = copy.copy(im)
        overlayed[road_mask == 1] = [255, 0, 0]

        # Save images into individual directories
        im_PIL = Image.fromarray(im)
        mask_PIL = Image.fromarray(road_mask.astype(np.uint8) * 255)
        overlayed_PIL = Image.fromarray(overlayed)
        im_PIL.save('output/resized/' + file_name + '.png')
        mask_PIL.save('output/map/' + file_name + '_m.png')
        overlayed_PIL.save('output/overlayed/' + file_name + '_o.png')
        im_PIL.save('output/all_together/' + file_name + '.png')
        mask_PIL.save('output/all_together/' + file_name + '_m.png')
        overlayed_PIL.save('output/all_together/' + file_name + '_o.png')