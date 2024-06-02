import cv2
import numpy as np

class GrayMeanFilter:
    def inverse_gamma_correction(self, value):
        if value <= 0.04045:
            return value / 12.92
        else:
            return ((value + 0.055) / 1.055) ** 2.4

    def gamma_correction(self, value):
        if value <= 0.0031308:
            return value * 12.92
        else:
            return 1.055 * (value ** (1 / 2.4)) - 0.055

    def rgb_to_grayscale(self, r, g, b):
        # 逆ガンマ補正
        r_prime = self.inverse_gamma_correction(r / 255.0)
        g_prime = self.inverse_gamma_correction(g / 255.0)
        b_prime = self.inverse_gamma_correction(b / 255.0)
        # 輝度計算
        v_prime = 0.2126 * r_prime + 0.7152 * g_prime + 0.0722 * b_prime
        # ガンマ補正
        v = self.gamma_correction(v_prime)

        return int(v * 255)

    def convert_image_to_grayscale_and_filter(self, input_image_path, output_filtered_image_path):
        # 画像の読み込み
        image = cv2.imread(input_image_path)
        # Mean Shiftフィルターを適用してノイズを除去
        filtered_image = cv2.pyrMeanShiftFiltering(image, sp=21, sr=51)
        # フィルタリングされたグレースケール画像を保存するための空の配列
        filtered_gray_image = np.zeros((filtered_image.shape[0], filtered_image.shape[1]), dtype=np.uint8)

        # 箱の中身はなんだろな？
        print(f'filtered_image:\n{filtered_image}\nfiltered_image.shape[0]:\n{filtered_image.shape[0]}\nfiltered_image.shape[1]:\n{filtered_image.shape[1]}')

        for i in range(filtered_image.shape[0]):
            for j in range(filtered_image.shape[1]):
                r, g, b = filtered_image[i, j]
                filtered_gray_image[i, j] = self.rgb_to_grayscale(r, g, b)

        # フィルター適用後のグレースケール画像の保存
        cv2.imwrite(output_filtered_image_path, filtered_gray_image)
        return filtered_gray_image
