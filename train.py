import torch
import torch.nn as nn
import utils


class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.frame_ldl_criterion = nn.KLDivLoss(reduction='none')
        self.ldl_criterion = nn.KLDivLoss()

    def forward(self, vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution, features, stored_info, _label_distribution, label, point_anno, cpc_loss, step):
        loss = {}

        loss_vid_ldl = self.ldl_criterion(torch.log(vid_distribution), _label_distribution).mean()     
           
        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).cuda()), dim=2)
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        # ce
        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()

        # soft bkg
        act_seed, bkg_seed = utils.select_seed_act_score(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())
        pos_num = (act_seed.max(dim=2)[0]>0).int().sum(dim=1).cuda()
        neg_num = ((bkg_seed>0).int().sum()).cuda()
        tot_num = pos_num+neg_num+1e-5
        rate = pos_num / neg_num
        act_seed = act_seed.cuda()
                
        # soft act
        weighting_p_act = act_seed
        num_p_actions = act_seed.max(dim=2)[0].sum(dim=1)
        
        if num_p_actions>0:
            focal_weight_p_act = (1 - cas_sigmoid_fuse) * (act_seed>0.5).int() + cas_sigmoid_fuse * (act_seed<0.5).int()
            focal_weight_p_act = focal_weight_p_act ** 2
            loss_frame_pact = 0.5 * (((focal_weight_p_act * self.ce_criterion(cas_sigmoid_fuse, act_seed) * weighting_p_act).sum(dim=2)).sum(dim=1) / num_p_actions).mean()
        else:
            loss_frame_pact = torch.zeros(1)[0].cuda()
            
        bkg_seed = bkg_seed.unsqueeze(-1).cuda()

        point_anno_bkg = torch.zeros_like(point_anno).cuda()
        point_anno_bkg[:,:,-1] = 1

        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        # ce 
        loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
        loss_total =  self.lambdas[0] * loss_vid_ldl + self.lambdas[1] * ((1-self.lambdas[2]) * loss_frame + self.lambdas[2] * (loss_frame_bkg + loss_frame_pact)) + self.lambdas[3] * cpc_loss

        loss['loss_recover_cpc'] = cpc_loss
        loss["loss_vid_ldl"] = loss_vid_ldl
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_frame_pact"] = loss_frame_pact        
        loss["loss_total"] = loss_total
        loss["pos_neg_rage"] = rate
        loss["pos_num"] = pos_num
        loss["neg_num"] = neg_num
        
        return loss_total, loss


def train_all(net, config, loader_iter, optimizer, criterion, logger, step):
    net.train()

    total_loss = {}
    total_cost = []

    optimizer.zero_grad()

    for _b in range(config.batch_size):

        _, _data, _label, _point_anno, stored_info, _, _, _label_distribution = next(loader_iter)

        _data = [_data[0].cuda(),_data[1].cuda()]
        _label = _label.cuda()
        _label_distribution = _label_distribution.cuda()
        _point_anno = _point_anno.cuda()

        vid_score, cas_sigmoid_fuse, cas_softmax_dis, features, vid_distribution, cpc_loss = net(_data, _label)

        cost, loss = criterion(vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution, features, stored_info, _label_distribution, _label, _point_anno, cpc_loss, step)

        total_cost.append(cost)

        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []

            if loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]

    total_cost = sum(total_cost) / config.batch_size

    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)
