import numpy as np
import matplotlib.pyplot as plt

images = np.load('images/images.npy')  # shape: (100, 1080, 1080, 6)
targets = np.load('images/targets.npy')  # shape: (100,)

#Inspect content of images and targets
print(images.shape, images.dtype)
print(targets.shape, targets.dtype)
print(np.unique(targets, return_counts=True))  # Class balance


def show_image(image, channel=0):
    plt.imshow(image[:, :, channel], cmap='gray')
    plt.axis('off')
    plt.show()



def extract_features(image):
    features = []
    for c in range(image.shape[-1]):
        channel = image[:, :, c]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.max(channel),
            np.min(channel)
        ])
    return features



show_image(images[0], channel=0)  # Visualize 1st image, 1st channel

X = np.array([extract_features(img) for img in images])
y = targets