import os
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# 加载模型
model = YOLO("/yolov10s.pt")  # 预训练的YOLOv8n模型

# 定义包含图像的文件夹路径
folder_path = '/home/ubuntu/04.det/yolov10-trt-c/old_img/'

# 定义输出文件夹，用于保存带有框的图像
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中图像文件的列表
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 按文件名中的数字顺序排序图像文件
# image_files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))

for image_file in tqdm(image_files):
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)
    
    # 运行推理
    results = model(img, stream=False, conf=0.3)  # 返回一个Results对象的生成器
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 将框转换为numpy数组并移动到CPU
        scores = result.boxes.conf.cpu().numpy()  # 置信度分数
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID
        
        # 处理每个检测到的框
        for box, score, class_id in zip(boxes, scores, class_ids):
            # 确保坐标在图像范围内
            x1, y1, x2, y2 = [max(0, min(coord, dim)) for coord, dim in zip(box, [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])]
            if x1 < x2 and y1 < y2:
                # 转换为Python浮点数
                x1, y1, x2, y2, score = map(float, [x1, y1, x2, y2, score])
                
                # 绘制边界框
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 准备带有类别ID和置信度的标签
                label = f"{class_id}: {score:.2f}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(y1, label_size[1])
                cv2.rectangle(img, (int(x1), int(top - label_size[1])), (int(x1 + label_size[0]), int(top + base_line)), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, label, (int(x1), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 保存带有边界框的图像
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, img)

print(f"带注释的图像已保存到 {output_folder}")