import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test_all import test
from model_tsl import *
from tensorboard_logger import Logger
from senti_features import *


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    # set hyperparameter
    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    # save config
    utils.save_config(config, os.path.join(config.output_path, "config.txt"))
    # initial network
    net = Model(config.len_feature, config.num_classes, config.r_act)
    net = net.cuda()
    # create dataloader
    test_loader = data.DataLoader(
        SentiFeature(data_path=config.data_path, mode='test',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=-1, sampling='random',
                        supervision='point', seed=config.seed),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)
    # log test results
    test_info = {"step": [],
                "average_mAP[0.1:0.3]": [], "average_nAP[0.1:0.3]": [],"average_pAP[0.1:0.3]": [], "mAP@0.10": [], "mAP@0.15": [], "mAP@0.20": [], "mAP@0.25": [], "mAP@0.30": [], "Rc@0.10": [], "Rc@0.20": [], "Rc@0.30": [], "Rc@0.15": [], "Rc@0.25": [], "F2@0.10": [], "F2@0.20": [], "F2@0.30": [], "F2@0.15": [], "F2@0.25": []}
    
    logger = Logger(config.log_path)

    net.load_state_dict(torch.load(args.model_file))

    test(net, config, logger, test_loader, test_info, 0)
    
    utils.save_best_record(test_info, 
        os.path.join(config.output_path, "best_record.txt"))
