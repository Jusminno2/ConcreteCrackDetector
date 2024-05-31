from grayscale_transform import GrayMeanFilter
from crack_detection import CrackDetection

def main():
    pic_name = input('Please enter the path of the image (without extension): ')
    input_image_path = f'pictures/{pic_name}.jpg'
    output_gray_image_path = f'pictures/o-{pic_name}.jpg'
    output_filtered_image_path = f'pictures/o-m-{pic_name}.jpg'
    output_binary_image_path = f'pictures/binary-{pic_name}.jpg'
    output_detected_image_path = f'pictures/detected-{pic_name}.jpg'

    gray_mean_filter = GrayMeanFilter()
    _, filtered_gray_image = gray_mean_filter.convert_image_to_grayscale_and_filter(input_image_path, output_gray_image_path, output_filtered_image_path)

    crack_detection = CrackDetection()
    crack_detection.detect_cracks(filtered_gray_image, output_detected_image_path, output_binary_image_path)

if __name__ == '__main__':
    main()
