import cv2
import numpy as np
from kittler_threshold import kittler_threshold

class CrackDetection:
    def compute_threshold(self, image):
        # Kittlerの方法で動的なしきい値を計算
        threshold = kittler_threshold(image)
        return threshold

    def compute_eccentricity(self, mask):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['mu20'] + moments['mu02'] == 0:
            return 0
        eccentricity = ((moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) / ((moments['mu20'] + moments['mu02']) ** 2)
        return np.sqrt(1 - eccentricity)

    def detect_cracks(self, filtered_gray_image, output_detected_image_path, output_binary_image_path):
        threshold_value = self.compute_threshold(filtered_gray_image)
        cracks = filtered_gray_image < threshold_value  # 二値化

        # 二値化された画像を保存
        binary_image = (cracks * 255).astype(np.uint8)
        cv2.imwrite(output_binary_image_path, binary_image)

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
