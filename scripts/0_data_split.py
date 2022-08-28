import os
from natsort import natsorted

root = '/home/ubuntu/data/sentiment/temporal/our_1219_full/vid'
for split in os.listdir(root):
    with open('/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/split_'+split+'.txt','w+')as txt:
        for vid in natsorted(os.listdir(os.path.join(root,split))):
            name, _ = os.path.splitext(vid)
            txt.write(name+'\n')