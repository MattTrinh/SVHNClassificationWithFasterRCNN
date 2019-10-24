# -*- coding: utf-8 -*-
import numpy as np
import h5py

TRAINING_DIR = "keras_rcnn/data/svhn/train"
TEST_DIR = "keras_rcnn/data/svhn/test"

def load_data():
    # Create training and test dictionaries
    training_dictionary = _generate_dictionary(TRAINING_DIR)
    test_dictionary = _generate_dictionary(TEST_DIR)
        
    return training_dictionary, test_dictionary

def _generate_dictionary(path):
    output_dict = {}
    file = h5py.File(path + "/digitStruct.mat", "r")
    bboxes = file["digitStruct"]["bbox"]
    names = file["digitStruct"]["name"]
    
    for image in range(len(names)):
        name_chars = file[names[0][0]]
        name = ''.join([char(name_chars[i]) for i in range(len(name_chars))])
    
        # Gather bounding boxes in an image
        img_bboxes = file[bboxes[0][image]]
        img_bboxes_data = []
        for n,v in bbox.items():
            img_bbox = []
            for i in v:
                img_bbox.append(int(file[i[0]][0][0]))
            img_bboxes.append(img_bbox)
        
        # Re-sort bounding box data into (x, y, w, h, label)
        img_bboxes_data = np.transpose(img_bboxes_data)
        resort = np.argsort([3, 4, 0, 1, 2])
        img_bboxes_data = img_bboxes_data[:,resort]
        
        # Add image and its bounding boxes to training dictionary
        output_dict[name] = img_bboxes_data
        
    return output_dict