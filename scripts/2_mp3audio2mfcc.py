from __future__ import print_function, division
import os, shutil
import sys
import subprocess
import json
import librosa
import numpy as np
from tqdm import tqdm
def class_process(img_path, dir_path, dst_dir_path):
    src_class_path = dir_path
    if not os.path.isdir(src_class_path):
        return
    dst_class_path = dst_dir_path
    if os.path.exists(dst_class_path):
        shutil.rmtree(dst_class_path)
    os.makedirs(dst_class_path)
    fpss = json.load(open('/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/fps_dict.json'))
    std = min(fpss.values())
    for file_name in tqdm(os.listdir(src_class_path)):
        if '.mp3' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        mfcc_file_name = name + '.npy'
        music_file_path = os.path.join(src_class_path, file_name)
        mfcc_file_path = os.path.join(dst_class_path, mfcc_file_name)
        fps = fpss[name]
        y, sr = librosa.load(music_file_path, sr=int(fps/std*44100/4.323882352941176*4/1.9992360580595874*2))
        # print(sr)
        #log mfcc
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
        logmelspec = librosa.power_to_db(melspectrogram)
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=96)
        # if logmelspec.shape[1]/len(os.listdir(os.path.join(img_path,name)))<2.:
        print(sr,logmelspec.shape[1]-2*len(os.listdir(os.path.join(img_path,name))))
        np.save(mfcc_file_path,logmelspec.T)
        

if __name__ == "__main__":
    # dir_path = sys.argv[1]  # avi directory
    # dst_dir_path = sys.argv[2]  # jpg directory
    split='test'
    img_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/pic/"+split
    dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/audio/"+split
    dst_dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/features/"+split+"/logmfcc"

    # for class_name in os.listdir(dir_path):
    #     class_process(dir_path, dst_dir_path, class_name)

    class_process(img_path, dir_path, dst_dir_path)
