import cv2
import numpy as np


class CrackDetection:
    def compute_threshold(self, image):
        crack_pixels = image[image < 127]
        non_crack_pixels = image[image >= 127]
        mean_crack = np.mean(crack_pixels)
        mean_non_crack = np.mean(non_crack_pixels)
        std_non_crack = np.std(non_crack_pixels)
        return min(mean_crack, mean_non_crack - 4 * std_non_crack)

    def compute_eccentricity(self, mask):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['mu20'] + moments['mu02'] == 0:
            return 0
        eccentricity = ((moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) / (
                    (moments['mu20'] + moments['mu02']) ** 2)
        return np.sqrt(1 - eccentricity)

    def detect_cracks(self, gray_image, filtered_gray_image, output_detected_image_path):
        threshold = self.compute_threshold(filtered_gray_image)
        cracks = gray_image < threshold
        num_labels, labeled_cracks = cv2.connectedComponents(cracks.astype(np.uint8))
        result_image = cv2.cvtColor(filtered_gray_image, cv2.COLOR_GRAY2BGR)

        for label in range(1, num_labels):
            mask = (labeled_cracks == label).astype(np.uint8)
            area = np.sum(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                eccentricity = self.compute_eccentricity(mask)
                if area >= 100 and area / (perimeter ** 2) <= 1 and eccentricity >= 0.97:
                    result_image[mask == 1] = [0, 0, 255]  # 赤色に設定

        cv2.imwrite(output_detected_image_path, result_image)
