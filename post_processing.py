import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class SegmentationPostProcessor:

    def predict(self, image, model):
        """
        Generate segmentation prediction and measure inference time.
        """

        start_time = time.time()

        prediction = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )[0, :, :, 0]

        inference_time = time.time() - start_time

        return prediction, inference_time

    def visualize(
            self,
            image,
            prediction,
            ground_truth,
            inference_time=None):
        """
        Visualize original image, predicted mask,
        and ground truth mask.
        """

        binary_mask = (
            prediction > 0.5
        ).astype(np.uint8)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap="gray")

        if inference_time is not None:
            plt.title(
                f"Prediction\n{inference_time:.4f} sec"
            )
        else:
            plt.title("Prediction")

        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def segmented_output(
            self,
            image,
            prediction):
        """
        Create segmented butterfly output.
        """

        binary_mask = (
            prediction > 0.5
        ).astype(np.uint8)

        segmented = (
            cv2.merge(
                [
                    binary_mask,
                    binary_mask,
                    binary_mask
                ]
            ) * image
        )

        return segmented

    def visualize_segmented_output(
            self,
            image,
            segmented_image):
        """
        Compare original and segmented output.
        """

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title("Segmented Butterfly")
        plt.axis("off")

        plt.tight_layout()
        plt.show()