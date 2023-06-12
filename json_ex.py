import json
import cv2 as cv
import time
with open('dataset/dsc_train/new.json') as file:
    json_file = json.load(file)

print(json_file)
uni_labels = set()
for e in json_file:
    print('name:')
    print(e['image'])
    print(e['annotations'][0]['coordinates'])
    uni_labels.add(e['annotations'][0]['label'])

print(uni_labels)