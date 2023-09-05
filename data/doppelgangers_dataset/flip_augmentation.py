""" Performs the flip augmentation for select files in the train_set_flip
image directory. Directly modifies the image directory.
"""
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing as mp
import shutil


##########################
# CONFIGURATION SETTINGS #
##########################

DATASET_PATH = './doppelgangers/images/train_set_noflip/'
""" Path to the base directory for train_set_noflip, fliptrain_set_flip, or train_megadepth images.
    Absolute path recommended.
"""

PAIRS_PATH = '../../../doppelgangers/pairs_metadata/train_pairs_noflip.npy'
""" Path to the pairs metadata file train_pairs_noflip.npy, train_pairs_flip.npy, or train_pairs_megadepth.npy.
    Absolute path recommended.
"""

ENABLE_MULTIPROCESSING = False
""" Whether to enable multiprocessing. """

THREAD_CT = 4
""" Number of processes to use if multiprocessing is enabled. A low number is
recommended since the image flip augmentation is memory intensive. """

#######################


def get_image_paths(pair_path):
    """ Returns a numpy array of image paths given a pair path. """
    pairs = np.load(pair_path, allow_pickle=True)

    return np.unique(
        np.array([[pair[0], pair[1]] for pair in pairs]).flatten())


def unflip_path(image_path):
    """ Return the unflipped image path of a flipped image path. """
    unflipped_path = image_path.replace('_flip', '')
    unflipped_path = unflipped_path.replace('flip_', '')
    unflipped_dir = os.path.dirname(unflipped_path)
    image_name_wo_ext = os.path.splitext(os.path.basename(unflipped_path))[0]

    # Find correct image extension of pre-flipped image
    try:
        for local_img in os.listdir("./{}".format(unflipped_dir)):
            if image_name_wo_ext in local_img:
                return "{}/{}".format(unflipped_dir, local_img)        
    except:
        pass

    # ... check dense subfolders
    try:
        for local_img in os.listdir("./{}/dense/images/".format(unflipped_dir)):
            if image_name_wo_ext in local_img:
                return "{}/dense/images/{}".format(unflipped_dir, local_img)
    except:
        pass

    for dense_folder_num in range(10):
        try:
            for local_img in os.listdir("./{}/dense{}/images/".format(unflipped_dir, dense_folder_num)):
                if image_name_wo_ext in local_img:
                    return "{}/dense{}/images/{}".format(unflipped_dir, dense_folder_num, local_img)
        except:
            pass

    raise Exception("Did not find unflipped variant of the flipped image path {}".format(image_path))


def save_flipped_image(flipped_path, overwrite=False):
    """ Locates the original unflipped image and saves the flipped image at the
    flipped path.
    """
    if not overwrite and os.path.isfile(flipped_path):
        return

    unflipped_path = unflip_path(flipped_path)

    original_image = Image.open(unflipped_path).convert('RGB')

    flipped_image = original_image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

    os.makedirs(os.path.dirname(flipped_path), exist_ok=True)
    flipped_image.save(flipped_path)


def do_flip_augmentation(pair_path, multithreading=True, threads=4):
    """ Flips the augments dataset at the current working directory based on the
    flipped images in pair_path.
    
    Precondition: must run os.chdir() before calling this function, such that
        the working directory for the script is at the root of the dataset.
        i.e. os.chdir('./doppelgangers/train_flip'). 
    """
    image_paths = get_image_paths(pair_path)

    print("Identifying images to flip...")
    flip_image_paths = []
    for image_path in tqdm(image_paths):
        origin_dir = os.path.dirname(image_path)
        if ('_flip' in origin_dir or 'flip_' in origin_dir):
            flip_image_paths.append(image_path)
    
    print("\nPerforming flip augmentations...")
    if multithreading:
        with mp.Pool(threads) as pool:
            pool.map(save_flipped_image, flip_image_paths)
    else:
        for image_path in tqdm(flip_image_paths):
            save_flipped_image(image_path)


def verify(pair_path):
    """ Verifys whether all images defined in the image pairs are present.
    
    Precondition: must run os.chdir() before calling this function, such that
        the working directory for the script is at the root of the dataset.
        i.e. os.chdir('./doppelgangers/train_flip'). 
    """
    image_paths = get_image_paths(pair_path)

    print("\nVerifying images...")
    for image_path in tqdm(image_paths):
        try:
            assert os.path.isfile(image_path)
        except Exception as e:
            print("ERROR: Cannot find {}".format(image_path))


if __name__ == '__main__':
    # Parameter changed to ensure all images can be processed.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Script initiation
    os.chdir(DATASET_PATH)
    do_flip_augmentation(PAIRS_PATH, multithreading=ENABLE_MULTIPROCESSING, threads=THREAD_CT)
    verify(PAIRS_PATH)