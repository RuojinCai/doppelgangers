# Doppelgangers Dataset

This is the Doppelgangers dataset, a benchmark dataset that allows for training and standardized evaluation of visual disambiguation algorithms. The dataset is described in the paper ["Doppelgangers: Learning to Disambiguate Images of Similar Structures"](https://doppelgangers-3d.github.io).

Doppelgangers consists of a collection of internet photos of world landmarks and cultural sites that exhibit repeated patterns and symmetric structures. The dataset includes a large number of image pairs, each labeled as either positive or negative based on whether they are true or false (illusory) matching pairs.

## One-step to download the complete dataset
We provide a `download.sh` script for downloading, extracting, and pre-processing the complete Doppelgangers Dataset. Dataset can be placed under the folder `./data/doppelgangers_dataset/` with the following layout:
<details>
<summary>[Click to expand]</summary>

```
|---doppelgangers
    |---images
        |---test_set
            |---...
        |---train_set_flip
            |---...
        |---train_set_noflip
            |---...
        |---train_megadepth
            |---...
    |---loftr_matches
        |---test_set
            |---...
        |---train_set_flip
            |---...
        |---train_set_noflip
            |---...
        |---train_megadepth
            |---...
    |---pairs_metadata
        |---...
```
</details>

If you want to download only a portion of the dataset, you can find detailed instructions below.

## Overview

This page includes downloads for:
* Train set without image flip augmentation
* Train set with image flip augmentation
* Test set
* COLMAP reconstructions
* Pretrained model checkpoints

For the train and test sets, we provide downloads for images, image pair labels, and precomputed LoFTR matches.

## Dataset Structure

### General File Structure
All compressed `tar.gz` files will extract into a joint `/doppelgangers/` directory. Generally, different types of content will map to the following subdirectories:
* Images &rarr; `/doppelgangers/images/(set_name)`
* Image pair info &rarr; `/doppelgangers/pairs_metadata/(set_name)`
* LoFTR matches &rarr; `/doppelgangers/loftr_matches/(set_name)`
* Pretrained models &rarr; `/doppelgangers/checkpoints/`
* COLMAP SfM reconstructions &rarr; `/doppelgangers/reconstructions/`

### Image File Structure
The image directory structure follows the WikiScenes dataset data structure as described in [Section 1, *Images and Textual Descriptions*](https://github.com/tgxs002/wikiscenes#the-wikiscenes-dataset).

### Image Pair Labels Structure
The image pair labels are stored using the NumPy `.npy` file format. There is one `.npy` file for every train or test set. Every `.npy` file contains a numpy array whose entries represent image pairs, and each entry is itself a numpy array.

An image pair entry has the format:
```
array([
    image_0_relative_path : str,
    image_1_relative_path : str,
    pos_neg_pair_label (pos=1, neg=0) : int,
    number_of_SIFT_matches : int
])
```

Example of an image pair entry:
```
array(['Berlin_Cathedral/east/0/pictures/Exterior of Berlin Cathedral 18.jpg',
       'Berlin_Cathedral/west/0/0/pictures/Exterior of Berlin Cathedral 14.jpg',
       0, 15], dtype=object)
```

### Precomputed LoFTR Matches Structure
The LoFTR matches are stored using the NumPy `.npy` file format. There are multiple `.npy` files per train or test set—one per image pair. The name of the `.npy` file is the index of the pair's location in the image pair NumPy array. 



## Downloads for Train Set, without Image Flip Augmentation
### Images and Precomputed Matches
* Images: [train_set_noflip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/train_set_noflip.tar.gz) (11G)
* LoFTR matches: [matches_train_noflip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_train_noflip.tar.gz) (1.2G)
* Image pair info: [(jump to section)](#downloads-for-image-pairs)

### Preparing the Dataset
Follow the [Preparing the Dataset](#preparing-the-dataset-1) section.

## Downloads for Train Set, with Image Flip Augmentation
### Images and Precomputed Matches
* Images:
  * Base images: [train_set_flip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/train_set_flip.tar.gz) (29G)
  * MegaDepth subset, images: [train_megadepth.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/train_megadepth.tar.gz) (41G)
  * MegaDepth subset, metadata: [megadepth.json](https://doppelgangers.cs.cornell.edu/dataset/megadepth.json)
  * Image flip augmentation script: [flip_augmentation.py](https://doppelgangers.cs.cornell.edu/dataset/flip_augmentation.py)
* LoFTR matches:
  * Base matches: [matches_train_flip.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_train_flip.tar.gz) (1.8G)
  * MegaDepth subset matches: [matches_megadepth.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_megadepth.tar.gz) (1.1G)
* Image pair info: [(jump to section)](#downloads-for-image-pairs)

### Preparing the Dataset
We provide a Python script `flip_augmentation.py` to perform the image flip augmentation on the provided base images. To use this script, please modify the configuration options at the beginning of the script and run with `python flip_augmentation.py`.

#### MegaDepth
This train set includes a subset of [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) images. Note that the MegaDepth images also have flip augmentations. Metadata on the subset of MegaDepth images that are used are stored in `megadepth.json`. The subset of images can also be directly downloaded, and are stored in `train_megadepth.tar.gz`.

Note that the file structure of our MegaDepth images are adjusted from the downloaded version. Let `xxxx` be the MegaDepth scene ID. The mapping from the download version to our file paths is as follows:
* The `xxxx/dense/images` in the downloaded version maps to our `xxxx/images/` directory.
* The `xxx/dense(int)/images` in the downloaded version maps to our `xxxx/images/dense(int)/images/` directory.
* The scene ID's `0147_1` and `0290_1` contain the `dense1` images of `0147` and `0290`, respectively. They are separated into a separate scene because the `dense1` images depict different landmarks from those depicted the original `dense` directories.

## Downloads for Test Set
### Images and Precomputed Matches
* Images: [test_set.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/test_set.tar.gz) (2G)
* LoFTR matches: [matches_test.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/matches_test.tar.gz) (76M)
* Image pair info: [(jump to section)](#downloads-for-image-pairs)

### Preparing the Dataset
No additional steps are required.

## Downloads for Image Pairs
Image pair metadata for all training and test sets: [pairs_metadata.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/pairs_metadata.tar.gz) (12M)


## Downloads for Pretrained Models
Pretrained model checkpoint with image flip augmentation: [checkpoint.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/checkpoint.tar.gz) (119M)


## Downloads for Reconstructions
[COLMAP](https://colmap.github.io/) reconstructions of the sixteen test scenes described in the paper: [reconstructions.tar.gz](https://doppelgangers.cs.cornell.edu/dataset/reconstructions.tar.gz) (3G)

## Dataset Attributions
Licensing information for images in the train and test sets sourced from Wikimedia Commons are here: [attributions.json](https://doppelgangers.cs.cornell.edu/dataset/attributions.json)


## Citation

If you find Doppelgangers useful for your work please cite:
```
@inproceedings{cai2023doppelgangers,
  title     = {Doppelgangers: Learning to Disambiguate Images of Similar Structures},
  author    = {Cai, Ruojin and Tung, Joseph and Wang, Qianqian and Averbuch-Elor, Hadar and Hariharan, Bharath and Snavely, Noah},
  journal   = {ICCV},
  year      = {2023}
}
```
