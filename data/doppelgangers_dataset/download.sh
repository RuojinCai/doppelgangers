echo Download and extract image pair metadata
wget -c https://doppelgangers.cs.cornell.edu/dataset/pairs_metadata.tar.gz
tar -xf pairs_metadata.tar.gz

echo Download and extract test set
wget -c https://doppelgangers.cs.cornell.edu/dataset/test_set.tar.gz
wget -c https://doppelgangers.cs.cornell.edu/dataset/matches_test.tar.gz

tar -xf test_set.tar.gz
tar -xf matches_test.tar.gz

echo Download and extract training set w/o image flip augmentation
wget -c https://doppelgangers.cs.cornell.edu/dataset/train_set_noflip.tar.gz
wget -c https://doppelgangers.cs.cornell.edu/dataset/matches_train_noflip.tar.gz

tar -xf train_set_noflip.tar.gz
tar -xf matches_train_noflip.tar.gz

echo Download and extract training set w/ image flip augmentation and MegaDepth subset
wget -c https://doppelgangers.cs.cornell.edu/dataset/train_set_flip.tar.gz
wget -c https://doppelgangers.cs.cornell.edu/dataset/train_megadepth.tar.gz
wget -c https://doppelgangers.cs.cornell.edu/dataset/matches_train_flip.tar.gz
wget -c https://doppelgangers.cs.cornell.edu/dataset/matches_megadepth.tar.gz

tar -xf train_set_flip.tar.gz
tar -xf train_megadepth.tar.gz
tar -xf matches_train_flip.tar.gz
tar -xf matches_megadepth.tar.gz

echo Prepare image flip augmentation data. It is recommended to provide absolute paths for the data in flip_augmentation.py
python flip_augmentation.py