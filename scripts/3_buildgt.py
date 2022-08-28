import os, shutil
import cv2
import numpy as np
import json
gt = {}
gt['database']={}
root = '/home/ubuntu/data/sentiment/temporal/our_1219_full/vid'
annoroot = '/home/ubuntu/data/sentiment/temporal/our_1219_full/label/full'
trainlist = os.listdir(os.path.join(root,'train'))

for split in ['train','test']:
    for vid in os.listdir(os.path.join(root,split)):
        name, _ = os.path.splitext(vid)
        annos = []
        # if split == 'test':
        anno_txt = np.loadtxt(os.path.join(annoroot,split,vid.split('_')[0]+'.txt'), dtype=str)
        if len(anno_txt.shape)!=0:
            for anno in anno_txt[1:]:
                s,e,cls = anno.split(',')
                annos.append({'segment':[s,e], 'label':'p' if cls=='positive' else 'n'})
        gt['database'][name]={'subset':'train' if vid in trainlist else 'test','annotations':annos}


with open("/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/videosenti_gt.json","w+") as f:
    json.dump(gt,f)