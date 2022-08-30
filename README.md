<div align="center">

# **Temporal Sentiment Localization**: Listen and Look in Untrimmed Videos


<i>Zhicheng Zhang and Jufeng Yang</i>

<a href=" "><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/ACM%20MM-2022-orange)](https://2022.acmmm.org/)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

</div>

<img src="./assests/fig0video.png" width="100%">

**Key motivation:** *a video may convey multiple sentiments and each sentiment appears with varying lengths and locations.* Images come from ***"The Wolf of Wall Street"***.

This repository contains the official implementation of our work in ACM MM 2022. TSL-300 dataset and pytorch training/validation code for **weakly-supervised framework TSL-Net** are released.

## Abstract

<img src="./assests/fig1motivation.png" width="50%" align="right">Video sentiment analysis aims to uncover the underlying attitudes of viewers, which has a wide range of applications in real world. Existing works simply classify a video into a single sentimental category, ignoring the fact that sentiment in untrimmed videos may appear in multiple segments with varying lengths and unknown locations. To address this, we propose a challenging task, *i.e.*, **T**emporal **S**entiment **L**ocalization (**TSL**), to find which parts of the video convey sentiment. To systematically investigate fully- and weakly-supervised settings for TSL, we first build a benchmark dataset named TSL-300, which is consisting of 300 videos with a total length of 1,291 minutes. Each video is labeled in two ways, one of which is frame-by-frame annotation for the fully-supervised setting, and the other is single-frame annotation, *i.e.*, only a single frame with strong sentiment is labeled per segment for the weakly-supervised setting. Due to the high cost of labeling a densely annotated dataset, we propose TSL-Net in this work, employing single-frame supervision to localize sentiment in videos. In detail, we generate the pseudo labels for unlabeled frames using a greedy search strategy, and fuse the affective features of both visual and audio modalities to predict the temporal sentiment distribution. Here, a reverse mapping strategy is designed for feature fusion, and a contrastive loss is utilized to maintain the consistency between the original feature and the reverse prediction. Extensive experiments  show the superiority of our method against the state-of-the-art approaches.

## Dependencies

You can set up the environments by using `pip3 install -r requirements.txt`.

#### Recommended Environment

* Python 3.6.13
* Pytorch 1.10.2
* CUDA 11.3

## TSL-300 dataset

If you need the TSL-300 dataset for academic purposes, please download the [application form](./assests/TSL-300_Data_Access_Form.docx) and fill out the request information, then send it to ***gloryzzc6@sina.com***.
We will process your application as soon as possible.
Please make sure that the email used comes from your educational institution.

## Data Preparation
1. Prepare [TSL-300](./assests/TSL-300_Data_Access_Form.docx) dataset.
    - We have provided constructed dataset and pre-extracted features.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used, which is also included in our dataset.
    - Link the features folder by using `sudo ln -s path-to-feature ./dataset/VideoSenti/`.
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.

~~~~
├── dataset
   └── VideoSenti
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── time.json
       ├── videosenti_gt.json
       ├── point_gaussian
           └── point_labels.csv
           ├── train
       └── features
           ├── train
               ├── rgb
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
               └── logmfcc
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
           └── test
               ├── rgb
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
               └── logmfcc
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
~~~~

## Model Zoo

The checkpoint files are coming soon.

<table>
    <tr>
        <td>Metric</td>
        <td>mAP@ 0.1</td>
        <td>mAP@ 0.2</td>
        <td>mAP@ 0.3</td>
        <td>mAP@AVG</td>
        <td>Recall@AVG</td>
        <td>F2@AVG</td>
        <td>Url</td>
    </tr>
    <tr>
        <td rowspan="2">TSL-Net</td>
        <td rowspan="2">27.27</td>
        <td rowspan="2">20.53</td>
        <td rowspan="2">12.06</td>
        <td rowspan="2">19.85</td>
        <td rowspan="2">75.24</td>
        <td rowspan="2">33.69</td>
        <td><a href="https://drive.google.com">Baidu drive</a></td>
    </tr>
    <tr><td><a href="https://drive.google.com">Google drive</a></td></tr>

</table>

## Running
You can easily train and evaluate the model by running the script below.


You can include more details such as epoch, batch size, etc. Please refer to `options.py`.

~~~~
$ bash run_train.sh
~~~~

## Evaulation
The pre-trained model can be found in [pretrained model](https://drive.google.com/file/d/1IgtUszVJjoa-VJ_UZVryFFcMouVJ__7d/view?usp=sharing).

You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [SF-Net](https://github.com/Flowerfan/SF-Net)
* [ActivityNet](https://github.com/activitynet/ActivityNet)
* [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)

## Citation

If you find this repo useful in your project or research, please consider citing the relevant publication.

````
@inproceedings{zhang2022temporal,
  title={Temporal Sentiment Localization: Listen and Look in Untrimmed Videos},
  author={Zhang, Zhicheng and Yang, Jufeng},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
````
