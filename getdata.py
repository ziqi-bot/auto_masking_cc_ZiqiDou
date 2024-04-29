import json

import numpy as np

path = 'imageNew/val'
# path = 'image/train' repeat the process for both train and val
import os

files = []
for file in os.listdir(path):
    if file[-5:] == '.json':
        files.append(file)

print(json.load(open('imageNew/val/' + files[0])))
#print(json.load(open('image/train/' + files[0])))

# Load annotations
# VGG Image Annotator (up to version 1.6) saves each image in the form:
# { 'filename': '28503151_5b5b7ec140_b.jpg',
#   'regions': {
#       '0': {
#           'region_attributes': {},
#           'shape_attributes': {
#               'all_points_x': [...],
#               'all_points_y': [...],
#               'name': 'polygon'}},
#       ... more regions ...
#   },
#   'size': 100202
# }

via_region_data = {}
for file in files:
    one_json = json.load(open('imageNew/val/' + file))
    #one_json = json.load(open('image/val/' + file))

    one_image = {}
    one_image['filename'] = file.split('.')[0] + '.jpg'
    shape = one_json['shapes']

    regions = {}
    for i in range(len(shape)):
        points = np.array(shape[i]['points'])
        all_points_x = list(points[:, 0])
        all_points_y = list(points[:, 1])

        regions[str(i)] = {}
        regions[str(i)]['region_attributes'] = {}
        regions[str(i)]['shape_attributes'] = {}

        regions[str(i)]['shape_attributes']['all_points_x'] = all_points_x
        regions[str(i)]['shape_attributes']['all_points_y'] = all_points_y
        regions[str(i)]['shape_attributes']['name'] = shape[i]['label']

    one_image['regions'] = regions
    one_image['size'] = 0

    via_region_data[file] = one_image
with open('imagesNew/val/via_region_data.json', 'w') as f:
#with open('images/val/via_region_data.json', 'w') as f:
    json.dump(via_region_data, f, sort_keys=False, ensure_ascii=True)

