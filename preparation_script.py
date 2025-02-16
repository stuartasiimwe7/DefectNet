import os
import cv2
import xml.etree.ElementTree as ET
import random
import shutil

#Paths
image_dir = "data/pcb_dataset/images/"
resized_dir = "data/pcb_dataset/resized_images/"
xml_dir = "data/pcb_dataset/Annotations/"
yolo_label_dir = "data/pcb_dataset/yolo_labels/"
train_image_dir = "data/pcb_dataset/train/images/"
val_image_dir = "data/pcb_dataset/val/images/"
train_label_dir = "data/pcb_dataset/train/labels/"
val_label_dir = "data/pcb_dataset/val/labels/"

#Classes
classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

#Create necessary directories
for dir_path in [resized_dir, yolo_label_dir, train_image_dir, val_image_dir, train_label_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)


#Step 1-Resize Images to 640x640
for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    output_class_dir = os.path.join(resized_dir, class_name)
    
    if os.path.isdir(class_path):
        os.makedirs(output_class_dir, exist_ok=True)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_resized = cv2.resize(img, (640, 640))
                cv2.imwrite(os.path.join(output_class_dir, img_name), img_resized)

print("Images resized to 640x640.")


#Step - Convert Pascal VOC Annotations to YOLO Format
def convert_voc_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find("filename").text
    image_size = root.find("size")
    img_w, img_h = int(image_size.find("width").text), int(image_size.find("height").text)

    yolo_label_path = os.path.join(output_dir, image_name.replace(".jpg", ".txt"))

    with open(yolo_label_path, "w") as yolo_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue
            
            class_idx = classes.index(class_name)
            bbox = obj.find("bndbox")

            x_min, y_min, x_max, y_max = map(int, [
                bbox.find("xmin").text, bbox.find("ymin").text, 
                bbox.find("xmax").text, bbox.find("ymax").text
            ])

            x_center = (x_min + x_max) / (2.0 * img_w)
            y_center = (y_min + y_max) / (2.0 * img_h)
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            yolo_file.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")


for class_name in os.listdir(xml_dir):
    class_dir = os.path.join(xml_dir, class_name)
    output_class_dir = os.path.join(yolo_label_dir, class_name)

    if os.path.isdir(class_dir):
        os.makedirs(output_class_dir, exist_ok=True)

        for xml_file in os.listdir(class_dir):
            if xml_file.endswith(".xml"):
                convert_voc_to_yolo(os.path.join(class_dir, xml_file), output_class_dir)

print("Pascal VOC annotations converted to YOLO format.")


#Step -Split Dataset into Train (80%) and Validation (20%) Sets
for class_name in classes:
    image_class_dir = os.path.join(resized_dir, class_name)
    label_class_dir = os.path.join(yolo_label_dir, class_name)

    if not os.path.exists(image_class_dir):
        continue

    images = [f for f in os.listdir(image_class_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    split_index = int(0.8 * len(images))

    for image_name in images[:split_index]:
        shutil.move(os.path.join(image_class_dir, image_name), os.path.join(train_image_dir, image_name))
        shutil.move(os.path.join(label_class_dir, image_name.replace('.jpg', '.txt')), os.path.join(train_label_dir, image_name.replace('.jpg', '.txt')))

    for image_name in images[split_index:]:
        shutil.move(os.path.join(image_class_dir, image_name), os.path.join(val_image_dir, image_name))
        shutil.move(os.path.join(label_class_dir, image_name.replace('.jpg', '.txt')), os.path.join(val_label_dir, image_name.replace('.jpg', '.txt')))

print("Dataset split into Train and Val.")
