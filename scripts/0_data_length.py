import os
import cv2
import json

root = '/home/ubuntu/data/sentiment/temporal/our_1219_full/vid'
numdurations={}

for split in os.listdir(root):
    for vid in os.listdir(os.path.join(root,split)):
        v = cv2.VideoCapture(os.path.join(root,split,vid))
        rate = v.get(5)
        cur_num_frame = v.get(7)
        cur_duration = cur_num_frame/rate
        name, _ = os.path.splitext(vid)
        numdurations[name]=cur_duration
with open("/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/len_duration_dict.json","w+") as f:
    json.dump(numdurations,f)