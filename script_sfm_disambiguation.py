import os
import yaml
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
import tqdm
import numpy as np

from doppelgangers.utils.process_database import create_image_pair_list, remove_doppelgangers
from doppelgangers.utils.loftr_matches import save_loftr_matches
from doppelgangers.utils.process_database import remove_doppelgangers


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    
    # colmap setting
    parser.add_argument('--colmap_exe_command', default='colmap', type=str,
                        help='colmap ext command')
    parser.add_argument('--matching_type', default='vocab_tree_matcher', type=str,
                        help="colmap feature matching type: ['vocab_tree_matcher', 'exhaustive_matcher']")
    parser.add_argument('--skip_feature_matching', default=False, action='store_true',
                        help="skip colmap feature matching stage")
    parser.add_argument('--database_path', default=None, type=str,
                        help="path to database.db")
    parser.add_argument('--skip_reconstruction', default=False, action='store_true',
                        help="skip colmap reconstruction w/o doppelgangers classifier")
    
    # input dataset setting
    parser.add_argument('--input_image_path', type=str,
                        help='path to input image dataset')

    # output setting
    parser.add_argument('--output_path', type=str,
                        help='path to output results')

    # Doppelgangers threshold setting
    parser.add_argument('--threshold', default=0.8, type=float,
                        help='doppelgangers threshold: smaller means more pairs will be included, larger means more pairs will be filtered out')

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--pretrained', default='weights/doppelgangers_classifier_loftr.pt', type=str,
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

def save_update_config(cfg, args):    
    # save a copy of updated config
    config_file = os.path.join(args.output_path, 'config.yaml')
    with open(args.config, 'r') as f:
        example_config = yaml.safe_load(f)
        
    example_config['data']['image_dir'] = cfg.data.image_dir
    example_config['data']['loftr_match_dir'] = cfg.data.loftr_match_dir
    example_config['data']['test']['pair_path'] = cfg.data.test.pair_path
    example_config['data']['output_path'] = cfg.data.output_path

    with open(config_file, "w") as f:
        yaml.dump(example_config, f)


def colmap_runner(args):
    if args.matching_type == 'vocab_tree_matcher':
        # efficient for large-scale scenes
        vocab_tree_path = 'weights/vocab_tree_flickr100K_words1M.bin'
        if not os.path.exists(vocab_tree_path):
            os.system('wget https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin')
            os.system('mv vocab_tree_flickr100K_words1M.bin weights/')
    
    # feature extraction
    command = [args.colmap_exe_command, 'feature_extractor', 
            '--image_path', args.input_image_path,
            '--database_path', args.database_path
            ]
    os.system(' '.join(command))
    
    # feature matching
    command = [args.colmap_exe_command, args.matching_type,
            '--database_path', args.database_path
            ]    
    if args.matching_type == 'vocab_tree_matcher':
        command += ['--VocabTreeMatching.vocab_tree_path', vocab_tree_path 
                ]            
    os.system(' '.join(command))


def doppelgangers_classifier(gpu, ngpus_per_node, cfg, args):
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
    gt_list = list()
    pred_list = list()
    prob_list = list()
    with torch.no_grad():
        for bidx, data in tqdm.tqdm(enumerate(test_loader)):
            data['image'] = data['image'].cuda()
            gt = data['gt'].cuda()
            score = decoder(data['image'])
            for i in range(score.shape[0]):
                prob_list.append(score[i].cpu().numpy())
                pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
                gt_list.append(gt[i].cpu().numpy())
    
    gt_list = np.array(gt_list).reshape(-1)
    pred_list = np.array(pred_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1, 2)    
    np.save(os.path.join(cfg.data.output_path, "pair_probability_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
    print("Test done.")


def main_worker(gpu, ngpus_per_node, cfg, args): 
    os.makedirs(args.output_path, exist_ok=True)      
    
    # colmap feature extraction and matching
    if not args.skip_feature_matching or args.database_path is None:
        print("colmap feature extraction and matching")
        args.database_path = os.path.join(args.output_path, 'database.db')
        colmap_runner(args)

    # extracting loftr matches
    print("Extracting loftr matches")
    loftr_matches_path = os.path.join(args.output_path, 'loftr_match')
    os.makedirs(loftr_matches_path, exist_ok=True)
    pair_path = create_image_pair_list(args.database_path, args.output_path)
    save_loftr_matches(args.input_image_path, pair_path, args.output_path)

    # edit config file with corresponding data path
    cfg.data.image_dir = args.input_image_path
    cfg.data.loftr_match_dir = loftr_matches_path
    cfg.data.test.pair_path = pair_path
    cfg.data.output_path = args.output_path    
    save_update_config(cfg, args)    
    
    # Running Doppelgangers classifier model on image pairs
    print("Running Doppelgangers classifier model on image pairs")
    doppelgangers_classifier(gpu, ngpus_per_node, cfg, args)

    # remove all the pairs with a probability lower than the threshold  
    print("remove all the pairs with a probability lower than the threshold in database")  
    pair_probability_file = os.path.join(args.output_path, "pair_probability_list.npy")
    update_database_path = remove_doppelgangers(args.database_path, pair_probability_file, pair_path, args.threshold)

    # colmap reconstruction with doppelgangers classifier  
    print("colmap reconstruction with doppelgangers classifier")  
    doppelgangers_result_path = os.path.join(args.output_path, 'sparse_doppelgangers_%.3f'%args.threshold)    
    os.makedirs(doppelgangers_result_path, exist_ok=True)       
    command = [args.colmap_exe_command, 'mapper',
           '--database_path', update_database_path,
           '--image_path', args.input_image_path,
           '--output_path', doppelgangers_result_path
          ]
    os.system(' '.join(command)) 

    # colmap reconstruction 
    if not args.skip_reconstruction:
        print("colmap reconstruction w/o doppelgangers classifier")  
        colmap_result_path = os.path.join(args.output_path, 'sparse')
        os.makedirs(colmap_result_path, exist_ok=True)
        command = [args.colmap_exe_command, 'mapper',
                '--database_path', args.database_path,
                '--image_path', args.input_image_path,
                '--output_path', colmap_result_path
                ]
        os.system(' '.join(command)) 


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