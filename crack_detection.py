import cv2
import numpy as np

class CrackDetection:
    def compute_eccentricity(self, mask):
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['mu20'] + moments['mu02'] == 0:
            return 0
        eccentricity = ((moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) / ((moments['mu20'] + moments['mu02']) ** 2)
        return np.sqrt(1 - eccentricity)

    def detect_cracks(self, filtered_gray_image, output_detected_image_path, output_binary_image_path):
        # 適応型しきい値処理を使用して二値化
        binary_image = cv2.adaptiveThreshold(filtered_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # 二値化された画像を保存
        cv2.imwrite(output_binary_image_path, binary_image)

        num_labels, labeled_cracks = cv2.connectedComponents(binary_image)

        result_image = cv2.cvtColor(filtered_gray_image, cv2.COLOR_GRAY2BGR)

        for label in range(1, num_labels):
            mask = (labeled_cracks == label).astype(np.uint8)
            area = np.sum(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                eccentricity = self.compute_eccentricity(mask)
                aspect_ratio = area / float(perimeter ** 2)
                if area >= 80 and eccentricity <= 0.97 and aspect_ratio <= 1.0:  # 条件を微調整
                    result_image[mask == 1] = [0, 0, 255]  # 赤色に設定

        cv2.imwrite(output_detected_image_path, result_image)

