import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

    def load_data(self):

        images = []
        masks = []

        image_names = os.listdir(self.image_dir)

        mask_names = [
            re.sub(r"\.png", "_seg0.png", name)
            for name in image_names
        ]

        for img_name, mask_name in zip(image_names, mask_names):

            try:

                img = plt.imread(
                    os.path.join(self.image_dir, img_name)
                )

                mask = plt.imread(
                    os.path.join(self.mask_dir, mask_name)
                )

            except FileNotFoundError:
                continue

            img = cv2.resize(
                img,
                (self.image_size, self.image_size)
            )

            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size)
            )

            images.append(img)
            masks.append(mask[:, :, 0])

        return np.array(images), np.array(masks)

    def split_data(self, test_size=0.2):

        images, masks = self.load_data()

        return train_test_split(
            images,
            masks,
            test_size=test_size,
            random_state=42
        )