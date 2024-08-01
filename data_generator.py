import os
import cv2
import numpy as np
import json
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_dir, annotation_dir, transforms, batch_size=8, image_size=448, shuffle=True):
        self.num_classes = 96
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index*self.batch_size:(index+1)*self.batch_size]
        images, bboxes, labels = self.__data_generation(batch_ids)
        return images, {'bboxes_output': bboxes, 'labels_output': labels}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generation(self, batch_ids):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        bboxes = np.zeros((self.batch_size, 4), dtype=np.float32)
        labels = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)

        for i, img_id in enumerate(batch_ids):
            img_path = os.path.join(self.image_dir, img_id + '.png')
            annotation_path = os.path.join(self.annotation_dir, img_id + '.json')
            image = cv2.imread(img_path)
            image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0

            with open(annotation_path) as f:
                annotation = json.load(f)

            bbox = annotation[0]['bbox']
            class_id = self.transforms[annotation[0]['class']]

            images[i] = image
            bboxes[i] = bbox
            labels[i][class_id] = 1

        return images, bboxes, labels