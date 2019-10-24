import hashlib
import json
import os.path
import shutil
import uuid

import numpy as np
import h5py
import pandas as pd
import cv2 as cv

def md5sum(pathname, blocksize=65536):
    checksum = hashlib.md5()

    with open(pathname, "rb") as stream:
        for block in iter(lambda: stream.read(blocksize), b""):
            checksum.update(block)

    return checksum.hexdigest()


def _generate_dictionary(path, num_images):
    output_dict = {}
    file = h5py.File(path + "/digitStruct.mat", "r")
    bboxes = file["digitStruct"]["bbox"]
    names = file["digitStruct"]["name"]
    
    for image in range(num_images):
        name_chars = file[names[image][0]]
        name = ''.join([chr(name_chars[i]) for i in range(len(name_chars))])
    
        # Gather bounding boxes in an image
        img_bboxes = file[bboxes[image][0]]
        img_bboxes_data = []
        for n,v in img_bboxes.items():
            img_bbox = []
            for i in v:
                if len(v) > 1:
                    img_bbox.append(int(file[i[0]][0][0]))
                else:
                    img_bbox.append(int(i[0]))
            img_bboxes_data.append(img_bbox)
        
        # Re-sort bounding box data into (x, y, w, h, label)
        img_bboxes_data = np.transpose(img_bboxes_data)
        resort = np.argsort([3, 4, 0, 1, 2])
        img_bboxes_data = img_bboxes_data[:,resort]
        
        # Add image and its bounding boxes to training dictionary
        output_dict[name] = img_bboxes_data
        
    file.close()
    return output_dict

def __main__():
    groups = ("train", "test")

    for group in groups:
        image_data = _generate_dictionary(group, 256) 
        directory = os.fsencode(group)
        dictionaries = []

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if not filename.endswith(".png"):
                continue
            
            pathname = os.path.join(group, filename)
            image = cv.imread(pathname, 0)
            r, c = image.shape[:2]

            if os.path.exists(pathname):
                dictionary = {
                    "image": {
                        "checksum": md5sum(pathname),
                        "pathname": pathname,
                        "shape": {
                            "r": r,
                            "c": c,
                            "channels": 3
                        }
                    },
                    "objects": []
                }

                for bbox in image_data[filename]:
                    minimum_r = int(bbox[0])
                    maximum_r = int(bbox[2] + minimum_r)
                    minimum_c = int(bbox[1])
                    maximum_c = int(bbox[3] + minimum_c)

                    object_dictionary = {
                        "bounding_box": {
                            "minimum": {
                                "r": minimum_r - 1,
                                "c": minimum_c - 1
                            },
                            "maximum": {
                                "r": maximum_r - 1,
                                "c": maximum_c - 1
                            }
                        },
                        "category": str(bbox[4])
                    }

                    dictionary["objects"].append(object_dictionary)

                dictionaries.append(dictionary)

        filename = "{}.json".format(group)

        with open(filename, "w") as stream:
            json.dump(dictionaries, stream)


if __name__ == "__main__":
    __main__()
