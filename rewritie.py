import json
import cv2
import os
from original_annotation import AnnotationReader
from natsort import natsorted
dataset_name = 'video_2023-04-05_22-42-06'
with open(f'dataset/{dataset_name}/new.json') as file:
    json_file = json.load(file)

new_lst = []
contents = os.listdir(f'dataset/{dataset_name}/')
# contents = [os.path.join('dsc_train/', f) for f in contents]
contents = natsorted(contents)
# contents = [e.split('/')[1] for e in contents]
cnt = 0
for content in contents:
    for e in json_file:
        if e['image'] == content:
            new_lst.append(e)
            cnt += 1
            break

print(cnt)
with open(f'dataset/{dataset_name}/new.json', 'w') as f:
    json.dump(new_lst, f)