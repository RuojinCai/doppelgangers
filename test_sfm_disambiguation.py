import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint
import re
import tqdm
import numpy as np


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config


def main_worker(gpu, ngpus_per_node, cfg, args):
    # basic setup
    cudnn.benchmark = True
    multi_gpu = False
    strict = True

    # initial dataset
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    test_loader = loaders['test_loader']

    # initial model
    decoder_lib = importlib.import_module(cfg.models.decoder.type)
    decoder = decoder_lib.decoder(cfg.models.decoder)
    decoder = decoder.cuda()

    # load pretrained model
    ckpt = torch.load(args.pretrained)
    import copy
    new_ckpt = copy.deepcopy(ckpt['dec'])
    if not multi_gpu:
        for key, value in ckpt['dec'].items():
            if 'module.' in key:
                new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)
    elif multi_gpu:
        for key, value in ckpt['dec'].items():                
            if 'module.' not in key:
                new_ckpt['module.'+key] = new_ckpt.pop(key)
    decoder.load_state_dict(new_ckpt, strict=strict)

    # evaluate on test set
    decoder.eval()
    acc = 0
    sum = 0
    gt_list = list()
    pred_list = list()
    prob_list = list()
    with torch.no_grad():
        for bidx, data in tqdm.tqdm(enumerate(test_loader)):
            data['image'] = data['image'].cuda()
            gt = data['gt'].cuda()
            score = decoder(data['image'])
            pred = torch.argmax(score,dim=1).cuda()
            acc += torch.sum(pred==gt).item()
            sum += score.shape[0]                
            for i in range(score.shape[0]):
                prob_list.append(score[i].cpu().numpy())
                pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
                gt_list.append(gt[i].cpu().numpy())
    
    gt_list = np.array(gt_list).reshape(-1)
    pred_list = np.array(pred_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1, 2)    
    np.save(os.path.join(cfg.data.output_path, "pair_probability_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
    print("Test done.")


def main():
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, cfg, args)


if __name__ == '__main__':
    main()