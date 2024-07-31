import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_bboxes(bbox_path):
    with open(bbox_path, 'r') as f:
        bboxes = json.load(f)
    return bboxes

def visualize_bboxes(image, bboxes):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb, cmap='gray')
    
    for bbox in bboxes:
        class_name = bbox['class']
        x1, y1, x2, y2 = bbox['bbox']
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1 - 10, class_name, color='red', fontsize=12, backgroundcolor='white')
    
    plt.show()

image_path = 'output/image_0.png'
bbox_path = 'output/image_0.json'

image = load_image(image_path)
bboxes = load_bboxes(bbox_path)
visualize_bboxes(image, bboxes)