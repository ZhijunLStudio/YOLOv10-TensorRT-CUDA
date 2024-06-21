import cv2
import numpy as np

def concatenate_images_with_spacing(img1_path, img2_path, output_path, spacing=50):
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 确保两张图片的高度相同
    height = max(img1.shape[0], img2.shape[0])
    width1 = img1.shape[1]
    width2 = img2.shape[1]

    if img1.shape[0] < height:
        img1 = cv2.copyMakeBorder(img1, 0, height - img1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if img2.shape[0] < height:
        img2 = cv2.copyMakeBorder(img2, 0, height - img2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 创建一个空白图像，用于在两张图片之间添加间隔
    blank_space = np.zeros((height, spacing, 3), dtype=np.uint8)

    # 拼接图片
    concatenated_img = np.hstack((img1, blank_space, img2))


    # 在每张图片上方写上文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    y_offset = 30

    cv2.putText(concatenated_img, 'yolov10s-pt-python', (width1 // 2 - 100, y_offset), font, font_scale, font_color, thickness)
    cv2.putText(concatenated_img, 'yolov10s-int8engine-c++', (width1 + width2 // 2 - 100, y_offset), font, font_scale, font_color, thickness)

    # 保存拼接后的图片
    cv2.imwrite(output_path, concatenated_img)

if __name__ == "__main__":
    img1_path = '/home/ubuntu/04.det/yolov10-trt-c/build/output_images/bus.jpg'
    img2_path = '/home/ubuntu/04.det/yolov10-trt-c/build/1.jpg'
    output_path = 'image1.jpg'

    concatenate_images(img1_path, img2_path, output_path)