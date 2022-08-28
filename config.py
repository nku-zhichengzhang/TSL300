import numpy as np
import os


class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr
        self.num_iters = len(self.lr)
        self.num_classes = 2
        self.modal = args.modal
        self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.log_path = args.log_path
        self.num_workers = args.num_workers
        self.lambdas = eval(args.lambdas)
        self.r_act = args.r_act
        self.class_thresh = args.class_th
        self.act_thresh_cas = np.arange(0, 0.25, 0.025)
        self.act_thresh_agnostic = np.arange(0.4, 0.75, 0.025)
        self.scale = 24
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 25
        
    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')


class_dict = {0: 'n',
                1: 'p'}