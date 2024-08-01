import os
import cv2
import numpy as np
import random
import json

data_dir = '../Query'

final_image_size = 448

symbol_size = 32

symbol_folders = [os.path.join(root, d) for root, dirs, _ in os.walk(data_dir) for d in dirs]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def add_noise(image, mean, var):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_dark_spots(image, num_spots=50, spot_intensity=50):
    row, col = image.shape
    for _ in range(num_spots):
        x = random.randint(0, col - 1)
        y = random.randint(0, row - 1)
        radius = random.randint(0, 1)
        image[y-radius:y+radius, x-radius:x+radius] = random.randint(0, spot_intensity)
    return image
def add_dashed_lines(image, num_lines=10, max_length=50):
    row, col = image.shape
    for _ in range(num_lines):
        x_start = random.randint(0, col - 1)
        y_start = random.randint(0, row - 1)
        length = random.randint(5, max_length)
        thickness = random.randint(1, 3)
        gap = random.randint(1, 5)
        color = random.choice([0, 255])
        orientation = random.choice(['horizontal', 'vertical'])

        if orientation == 'horizontal':
            x_end = min(col, x_start + length)
            for x in range(x_start, x_end, gap + thickness):
                cv2.line(image, (x, y_start), (min(col, x + thickness), y_start), color, 1)
        else:
            y_end = min(row, y_start + length)
            for y in range(y_start, y_end, gap + thickness):
                cv2.line(image, (x_start, y), (x_start, min(row, y + thickness)), color, 1)

    return image

def add_curved_lines(image, num_lines=10, max_points=100, max_distance=70):
    row, col = image.shape
    for _ in range(num_lines):
        start_x = random.randint(0, col)
        start_y = random.randint(0, row)
        end_x = random.randint(max(0, start_x - max_distance), min(col, start_x + max_distance))
        end_y = random.randint(max(0, start_y - max_distance), min(row, start_y + max_distance))
        curvature = random.uniform(-0.5, 0.5)
        color = random.choice([0, 255])
        thickness = random.randint(1, 3)

        points = []
        for t in np.linspace(0, 1, max_points):
            x = int((1 - t) * start_x + t * end_x + curvature * t * (1 - t) * col / 10)
            y = int((1 - t) * start_y + t * end_y + curvature * t * (1 - t) * row / 10)
            points.append([x, y])

        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, color, thickness)
    return image

def is_overlapping(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)
def create_image_with_symbols(symbol_folders, num_symbols=10):
    final_image = np.ones((final_image_size, final_image_size), np.uint8) * 255
    bounding_boxes = []

    for _ in range(num_symbols):
        folder = random.choice(symbol_folders)
        images = load_images_from_folder(folder)
        if not images:
            continue
        
        symbol_img = random.choice(images)
        
        symbol_img = cv2.resize(symbol_img, (symbol_size, symbol_size))
        
        symbol_img = add_noise(symbol_img, 30, 10)

        attempts_limit = 100
        attempts = 0
        placed = False
        while attempts < attempts_limit and not placed:
            max_x = final_image_size - symbol_size
            max_y = final_image_size - symbol_size
            top_left_x = random.randint(0, max_x)
            top_left_y = random.randint(0, max_y)

            new_bbox = [top_left_x, top_left_y, top_left_x + symbol_size, top_left_y + symbol_size]

            if all(not is_overlapping(new_bbox, bbox["bbox"]) for bbox in bounding_boxes):
                final_image[top_left_y:top_left_y + symbol_size, top_left_x:top_left_x + symbol_size] = symbol_img
                bounding_boxes.append({
                    "class": folder.split('_')[-1] + folder.split('_')[-2],
                    "bbox": new_bbox
                })
                placed = True
            attempts += 1

        if not placed:
            print(f"Не удалось разместить символ после {attempts_limit} попыток.")

    final_image = add_noise(final_image, -50, 200)

    final_image = add_dark_spots(final_image, num_spots=1000, spot_intensity=10)
    final_image = add_dark_spots(final_image, num_spots=500, spot_intensity=30)
    
    final_image = add_dashed_lines(final_image, num_lines=15, max_length=100)
    final_image = add_curved_lines(final_image, num_lines=10)

    return final_image, bounding_boxes

def save_image_with_bboxes(image, bboxes, image_path, bbox_path):
    cv2.imwrite(image_path, image)
    with open(bbox_path, 'w') as f:
        json.dump(bboxes, f)

num_samples = 2600
output_dir = './output/train'

if not os.path.exists(output_dir):
    raise "Такой директории нет"

for i in range(num_samples):
    image, bboxes = create_image_with_symbols(symbol_folders, 20)
    image_path = os.path.join(output_dir, f'image_{400 + i}.png')
    bbox_path = os.path.join(output_dir, f'image_{400 + i}.json')
    save_image_with_bboxes(image, bboxes, image_path, bbox_path)