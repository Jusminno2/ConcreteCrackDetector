from grayscale_transform import ImageProcessor

def main():
    # 入力画像と出力画像のパス
    pic_name = input('Please enter the path of the image (without extension): ')
    input_image_path = f'pictures/{pic_name}.jpg'
    output_gray_image_path = f'pictures/o-{pic_name}.jpg'
    output_filtered_image_path = f'pictures/o-m-{pic_name}.jpg'

    # 画像のグレースケール変換とノイズ除去
    processor = ImageProcessor()
    processor.convert_image_to_grayscale(input_image_path, output_gray_image_path, output_filtered_image_path)

if __name__ == '__main__':
    main()
