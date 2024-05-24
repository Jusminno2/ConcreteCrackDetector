import cv2
import numpy as np


def inverse_gamma_correction(value):
    if value <= 0.04045:
        return value / 12.92
    else:
        return ((value + 0.055) / 1.055) ** 2.4


def gamma_correction(value):
    if value <= 0.0031308:
        return value * 12.92
    else:
        return 1.055 * (value ** (1 / 2.4)) - 0.055


def rgb_to_grayscale(r, g, b):
    # 逆ガンマ補正
    r_prime = inverse_gamma_correction(r / 255.0)
    g_prime = inverse_gamma_correction(g / 255.0)
    b_prime = inverse_gamma_correction(b / 255.0)

    # 輝度計算
    v_prime = 0.2126 * r_prime + 0.7152 * g_prime + 0.0722 * b_prime

    # ガンマ補正
    v = gamma_correction(v_prime)

    return int(v * 255)


def convert_image_to_grayscale(input_image_path, output_gray_image_path, output_filtered_image_path):
    # 画像の読み込み
    image = cv2.imread(input_image_path)

    # 出力用の空の画像
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 画像の各ピクセルをグレースケールに変換
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            gray_image[i, j] = rgb_to_grayscale(r, g, b)

    # グレースケール画像の保存（フィルター適用前）
    cv2.imwrite(output_gray_image_path, gray_image)

    # Medianフィルターを適用してノイズを除去
    filtered_image = cv2.medianBlur(gray_image, 5)  # カーネルサイズを5に設定

    # フィルター適用後のグレースケール画像の保存
    cv2.imwrite(output_filtered_image_path, filtered_image)


# 入力画像と出力画像のパス
input_image_path = 'pictures/sample03.jpg'
output_gray_image_path = 'pictures/o-sample03.jpg'
output_filtered_image_path = 'pictures/o-m-sample03.jpg'

# 画像のグレースケール変換とノイズ除去
convert_image_to_grayscale(input_image_path, output_gray_image_path, output_filtered_image_path)

# img = cv2.imread('pictures/sample02.jpg', 0)
#
# area = [3, 15, 31, 63, 127, 255]
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     new_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, area[i], 0)
#     plt.imshow(new_img, 'gray')
#     plt.title("block size is {}".format(area[i]))
#     plt.xticks([]), plt.yticks([])
#
# plt.show()
