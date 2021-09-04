import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import csv
import base64
import json_lines
import lmdb # install lmdb by "pip install lmdb"
import pickle
csv.field_size_limit(sys.maxsize)

def converId(img_id, c):

    img_id = img_id.split('-')
    if 'train' in img_id[0]:
        new_id = int(img_id[1])
    elif 'val' in img_id[0]:
        if c == 1:
            new_id = int(img_id[1]) + 1000000
        else:
            new_id = int(img_id[1]) + 3000000
    elif 'test' in img_id[0]:
        new_id = int(img_id[1]) + 2000000    
    else:
        pdb.set_trace()

    return new_id

image_path = {}
path2id = {}
metadata_path = {}
with open('../X_VCR/train.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            img_id = item['img_id']
            metadata_path[item['metadata_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'], 0)

with open('../X_VCR/val.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'], 1)
            metadata_path[item['metadata_fn']] = 1

with open('../X_VCR/test.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'], 2)
            metadata_path[item['metadata_fn']] = 1

with open('../X_VCR/orig_val.jsonl', 'rb') as f: # opening file in binary(rb) mode
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'], 3)
            metadata_path[item['metadata_fn']] = 1

count = 0
num_file = 1
name = 'feature/VCR/VCR_gt_resnet101_faster_rcnn_genome.tsv.%d'
infiles = [name % i for i in range(num_file)]
length = len(image_path)
print("total length is %d" %length)

id_list = []
save_path = os.path.join('VCR_gt_resnet101_faster_rcnn_genome.lmdb')
save_path_ids = os.path.join('image_id_gt.lmdb')
env = lmdb.open(save_path, map_size=1099511627776)
with env.begin(write=True) as txn:
    s = 0
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                img_id = str(path2id[item['image_id']]).encode()
                if len(str(path2id[item['image_id']])) == 7 and str(path2id[item['image_id']])[0] == '1':
                    s += 1
                    # print(s, count, item['image_id'], str(path2id[item['image_id']]))
                txn.put(img_id, pickle.dumps(item))
                id_list.append(img_id)
                # print(item.keys())
                if count % 1000 == 0:
                    print(count)
                count += 1
                
    print(s)
    print(txn.get('1000209'.encode()) is not None)
                
env_id = lmdb.open(save_path_ids, map_size=1099511627776)
with env_id.begin(write=True) as txn_id:
    txn_id.put('keys'.encode(), pickle.dumps(id_list))
    
print(count)
json.dump(path2id, open('VCR_gt_ImagePath2Id.json', 'w'))
