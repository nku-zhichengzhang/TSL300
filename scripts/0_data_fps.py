import os
import cv2
import json

root = '/home/ubuntu/data/sentiment/temporal/our_1219_full/vid'
fps={}

for split in os.listdir(root):
    for vid in os.listdir(os.path.join(root,split)):
        v = cv2.VideoCapture(os.path.join(root,split,vid))
        cur_fps = v.get(cv2.CAP_PROP_FPS)
        name, _ = os.path.splitext(vid)
        fps[name]=cur_fps
with open("/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/fps_dict.json","w+") as f:
    json.dump(fps,f)