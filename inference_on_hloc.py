from types import SimpleNamespace

from tqdm import tqdm
from doppelgangers.models.cnn_classifier import decoder
from doppelgangers.datasets.hloc_dataset import HlocDoppelgangersDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import h5py
import argparse

def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    
    # colmap setting
    parser.add_argument('--weights_path', type=str,
                        help="Path to classifier weights")
    parser.add_argument('--features_path', type=str,
                        help="path to hloc features HDF5 file")
    parser.add_argument('--matches_path', type=str,
                        help="path to hloc matches HDF5 file")
    parser.add_argument('--filtered_path', type=str,
                        help="path to hloc matches HDF5 file")
    parser.add_argument('--image_dir', type=str,
                        help="path to where images are stored")
    parser.add_argument('--pairs_path', type=str,
                        help="pairs_path path to sfm-pairs.txt")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size")

    args = parser.parse_args()
    return args


args = get_args()
weights_path = args.weights_path
features_file = args.features_path
matches_file = args.matches_path
sfm_filtered = args.filtered_path
image_dir = args.image_dir
pair_path = args.pairs_path
batch_size = args.batch_size

multi_gpu = False
model = decoder(cfg=SimpleNamespace(input_dim=10))
ckpt = torch.load(weights_path)
new_ckpt = copy.deepcopy(ckpt['dec'])

for key, value in ckpt['dec'].items():
    if 'module.' in key:
        new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)

model.load_state_dict(new_ckpt, strict=True)
model = model.cuda().eval()

# weights_path = '/files/weights/doppelgangers_classifier_loftr.pt'
# features_file='/outputs/features.h5'
# matches_file='/outputs/matches.h5'
# sfm_filtered='/outputs/pairs-sfm-filtered.txt'
# image_dir='/images',
# pair_path='/pairs-sfm.txt',

with h5py.File(features_file, 'r') as features_f,  h5py.File(matches_file, 'r') as matches_f,  open(sfm_filtered, 'w') as filterd_f:
    test_loader = DataLoader(
        dataset=HlocDoppelgangersDataset(
            img_size=640,
            image_dir=image_dir,
            pair_path=pair_path,
            features_file=features_f,
            matches_file=matches_f
        ),
        batch_size=batch_size,
        shuffle=False, num_workers=8, drop_last=False)

    for b in tqdm(test_loader):
        with torch.no_grad():
            scores = model(b['image'].cuda()).detach().cpu().numpy()
            for i1, i2, score in zip(b['image1_name'], b['image2_name'], scores):
                score_argmax = np.argmax(score)
                filterd_f.write(f"{i1} {i2} {score_argmax}\n")
        
