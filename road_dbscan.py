import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.segmentation import flood_fill
from skimage.morphology import binary_closing
import glob
from tqdm import tqdm
import os
import copy

class RoadDetection:
    def __init__(self, im: np.ndarray):
        self.im = im
        self.processed_im = self.__preprocess(im)
        self.road_mask = self.__segment_road(self.processed_im)
    
    def __preprocess(self, im: np.ndarray) -> np.ndarray:
        # Crop
        im_crop = im[im.shape[0] // 3:, :]

        # Use histogram equalization to increase the image's contrast
        # im_eq = np.array([exposure.equalize_hist(im_crop[..., 0]), exposure.equalize_hist(im_crop[..., 1]), exposure.equalize_hist(im_crop[..., 2])])
        im_eq = exposure.equalize_hist(im_crop.reshape(-1)).reshape(im_crop.shape)

        # Blur the image to make the road look more even
        im_blur = gaussian_filter(im_eq, sigma=3)
        return im_blur

    def __get_largest_label(self, labels: np.ndarray) -> int:
        uniques = np.unique(labels)
        sizes = []
        for u in uniques:
            sizes.append(len(labels[labels==u]))
        size = np.sort(sizes)[-2]
        ind_max = np.argwhere(sizes == size)
        return uniques[ind_max].reshape(-1)[0]

    def __gaussian2D(self, height: int, width: int, sigma_x: float, sigma_y: float) -> np.ndarray:
        h_range = height / width
        w_range = 1
        x = np.linspace(-w_range, w_range, width)
        y = np.linspace(-h_range, h_range, height)
        x, y = np.meshgrid(x, y)
        z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
        + y**2/(2*sigma_y**2))))
        z_norm = z / np.sum(z)
        return z_norm

    def __diamond(self, n):
        a = np.arange(n)
        b = np.minimum(a,a[::-1])
        return ((b[:,None]+b)>=(n-1)//2).astype(np.int)
    
    def __segment_road(self, im: np.ndarray) -> np.ndarray:
        # Reshape the image into NxD array
        im_flat = im.reshape(-1, 3)
        # This paper suggests choosing epsilon based on the distance of the 3rd nearest neighbor
        # https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdfhttps://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
        nbrs = NearestNeighbors(n_neighbors=3).fit(im_flat)
        distances, indices = nbrs.kneighbors(im_flat)
        distances = distances[:, 2]
        distances = np.sort(distances, axis=0)
        eps = distances[-3]

        # Use DBSCAN clustering algorithm
        clustering = DBSCAN(eps=eps, min_samples=40).fit(im_flat)
        labels = clustering.labels_

        # Find label that has a lot of its points in the middle of the image
        unique_labels = np.unique(labels)
        label_total_prob = []
        gaussian2D = self.__gaussian2D(im.shape[0], im.shape[1], sigma_x=0.5, sigma_y=0.3).reshape(-1)
        for label in unique_labels:
            im_cluster = np.zeros(im_flat.shape[0], dtype='float64')
            im_cluster[labels == label] = 1
            prob_2d = im_cluster * gaussian2D
            total_prob = np.sum(prob_2d)
            label_total_prob.append(total_prob)
        ind_max = np.argmax(label_total_prob)
        max_label = unique_labels[ind_max]

        im_clustered = np.zeros(im_flat.shape[0], dtype='float64')
        im_clustered[labels == max_label] = 1
        im_clustered = im_clustered.reshape(self.processed_im.shape[:2])

        # Find the connected components
        all_labels = measure.label(im_clustered, background=0)

        # Find the largest connected component and make a road mask
        largest_label = self.__get_largest_label(all_labels)
        mask = np.zeros(all_labels.shape)
        mask[all_labels == largest_label] = 1

        # Assume the road fills from the bottom of the image
        # for c in range(mask.shape[1]):
        #     r = mask.shape[0] - 1
        #     if mask[r, c] == 0:
        #         mask = flood_fill(mask, (r, c), 1)
        # mask = flood_fill(mask, (mask.shape[0] - 1, 0), 1)

        # Use binary closing to fill in small gaps on the road
        kernel = self.__diamond(17)
        closed = binary_closing(mask, kernel)

        # Put the segmentation on an image of the original size
        im_uncrop = np.zeros(self.im.shape[:2])
        im_uncrop[(self.im.shape[0] // 3):, :] = closed
        return im_uncrop

# Load and resize image
def load_image(filepath: str) -> np.ndarray:
    im: Image.Image = Image.open(filepath)
    (width, height) = (im.width // 4, im.height // 4)
    im_resized = im.resize((width, height))
    return np.array(im_resized)

image_names = glob.glob('all_images/img*.png')

for image in tqdm(image_names):
# for image in image_names:
    # File name without .png or parent directories
    file_name = os.path.basename(image)[:-4]
    
    im = load_image(image)

    # Run road detection algorithm
    im_RoadData = RoadDetection(im)

    # Get the road mask
    bw_map = im_RoadData.road_mask
    plt.imshow(bw_map)
    plt.show()
    
    # Copy the base image and get a version of it with the road mask layered on top
    map_overlayed = copy.copy(im)
    map_overlayed[bw_map == 1] = [255, 0, 0]

    # Save images into individual directories
    im_PIL = Image.fromarray(im)
    bw_map_PIL = Image.fromarray(bw_map.astype(np.uint8) * 255)
    map_overlayed_PIL = Image.fromarray(map_overlayed)
    im_PIL.save('output/resized/' + file_name + '.png')
    bw_map_PIL.save('output/map/' + file_name + '_m.png')
    map_overlayed_PIL.save('output/overlayed/' + file_name + '_o.png')
    im_PIL.save('output/all_together/' + file_name + '.png')
    bw_map_PIL.save('output/all_together/' + file_name + '_m.png')
    map_overlayed_PIL.save('output/all_together/' + file_name + '_o.png')