import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import train_all, Total_loss
from test_all import test
from model_tsl import *
import tensorboardX
from tensorboard_logger import Logger
from senti_features import *
from tqdm import tqdm


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
    train_loader = data.DataLoader(
        SentiFeature(data_path=config.data_path, mode='train',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=-1, sampling='random',
                        supervision='point', seed=config.seed),
            batch_size=1,
            shuffle=True, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

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
                "average_mAP[0.1:0.3]": [], "average_nAP[0.1:0.3]": [],"average_pAP[0.1:0.3]": [],
                "mAP@0.10": [], "mAP@0.15": [], "mAP@0.20": [], "mAP@0.25": [], "mAP@0.30": [],
                "Rc@0.10": [], "Rc@0.20": [], "Rc@0.30": [], "Rc@0.15": [], "Rc@0.25": [],
                "F2@0.10": [], "F2@0.20": [], "F2@0.30": [], "F2@0.15": [], "F2@0.25": []}

    best_mAP = -1
    # create loss
    criterion = Total_loss(config.lambdas)

    # build optimizer
    a_params = list(map(id, net.cls_module.a_extractor.parameters()))
    base_params = filter(lambda p: id(p) not in a_params, net.parameters())
    optimizer = torch.optim.Adam([{'params': base_params},
                                {'params': net.cls_module.a_extractor.parameters(), 'lr':10*config.lr[0]}],
                                lr=config.lr[0], betas=(0.9, 0.999), weight_decay=0.0005)
    # intial logger
    logger = Logger(config.log_path)
    
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        # lr update
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        
        if (step - 1) % (len(train_loader) // config.batch_size) == 0:
            loader_iter = iter(train_loader)
        # train a iteration
        train_all(net, config, loader_iter, optimizer, criterion, logger, step)
        # test
        if step % 100 == 0:      
            test(net, config, logger, test_loader, test_info, step)
            for name, layer in net.named_parameters():
                if layer.requires_grad == True and layer.grad is not None:
                    logger.log_histogram(name + '_grad', layer.grad.cpu().data.numpy(), step)
                    logger.log_histogram(name + '_data', layer.cpu().data.numpy(), step)

            if test_info["average_mAP[0.1:0.3]"][-1] > best_mAP:
                best_mAP = test_info["average_mAP[0.1:0.3]"][-1]
                # save test results
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "best_record_seed_{}.txt".format(config.seed)))

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "model_seed_{}.pkl".format(config.seed)))