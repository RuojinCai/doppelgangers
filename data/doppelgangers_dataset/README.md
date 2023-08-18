# Doppelgangers Dataset

This is the Doppelgangers dataset, a benchmark dataset that allows for training and standardized evaluation of visual disambiguation algorithms. The dataset is described in the paper ["Doppelgangers: Learning to Disambiguate Images of Similar Structures"](https://doppelgangers-3d.github.io).

Doppelgangers consists of a collection of internet photos of world landmarks and cultural sites that exhibit repeated patterns and symmetric structures. The dataset includes a large number of image pairs, each labeled as either positive or negative based on whether they are true or false (illusory) matching pairs.

## Overview

This page includes downloads for:
* Train set without image flip augmentation
* Train set with image flip augmentation
* Test set
* COLMAP reconstructions
* Pretrained model checkpoints

### Train and Test Sets
We provide separate downloads for images, image pair labels, and precomputed LoFTR matches.

## Dataset Structure

### Images
The image directory structure follows the WikiScenes dataset data structure as described in [Section 1, *Images and Textual Descriptions*](https://github.com/tgxs002/wikiscenes#the-wikiscenes-dataset).

The image pair labels use the NumPy `.npy` file format.


### Post-Extraction File Structure
All compressed `tar.gz` files will extract into a joint `/doppelgangers/` directory. Generally, these types of content will map to the following subdirectories:
* Images &rarr; `/doppelgangers/images/(set_name)`
* Image pair info &rarr; `/doppelgangers/pairs_metadata/(set_name)`
* LoFTR matches &rarr; `/doppelgangers/loftr_matches/(set_name)`
* Pretrained models &rarr; `/doppelgangers/checkpoints/`
* COLMAP SfM reconstructions &rarr; `/doppelgangers/reconstructions/`



## Dataset Downloads
### Images and Precomputed Matches
#### Train Set, without Flip Augmentation
* Images: [train_set_noflip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/train_set_noflip.tar.gz)
* LoFTR matches: [matches_train_noflip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_train_noflip.tar.gz)

**Preparing the dataset:** No additional steps are required.

#### Train Set, with Flip Augmentation
* Images:
  * Base images: [train_set_flip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/train_set_flip.tar.gz)
  * Flip augmentation script: [flip_augmentation.py](https://doppelgangers.cs.cornell.edu/dataset/flip_augmentation.py)
  * Additional images, MegaDepth augmentation: [megadepth.json](https://doppelgangers.cs.cornell.edu/dataset/megadepth.json)
* LoFTR matches:
  * Base matches: [matches_train_flip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_train_flip.tar.gz)
  * Additional matches, MegaDepth augmentation: [matches_megadepth.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_megadepth.tar.gz)

**Preparing the dataset:** We provide a Python script `flip_augmentation.py` to perform the image flip augmentation on the provided base images. 

To use this script, please modify the configuration options at the beginning of the script and run with `python flip_augmentation.py`.

#### Test Set
* Images: [test_set.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/test_set.tar.gz)
* LoFTR matches: [matches_test.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_test.tar.gz)

**Preparing the dataset:** No additional steps are required.

### Image Pairs
* Image pairs information for all training and test sets: [pairs_metadata.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_test.tar.gz)


## Pretrained Models
The pretrained model with image flip augmentation: [checkpoint.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/checkpoint.tar.gz)


## Reconstructions
[COLMAP](https://colmap.github.io/) reconstructions of the sixteen test scenes described in the paper: [reconstructions.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/reconstructions.tar.gz)

## Dataset Attributions
Licensing information for images in the train and test sets sourced from Wikimedia Commons are here: [attributions.json](https://doppelgangers.cs.cornell.edu/dataset/attributions.json)


## Citation

If you find Doppelgangers useful for your work please cite:
```
@inproceedings{cai2023doppelgangers,
  title     = {Doppelgangers: Learning to Disambiguate Images of Similar Structures},
  author    = {Cai, Ruojin and Tung, Joseph and Wang, Qianqian and Averbuch-Elor, Hadar and Hariharan, Bharath and Snavely, Noah},
  journal   = {ICCV},
  year      = {2023},
}
```
