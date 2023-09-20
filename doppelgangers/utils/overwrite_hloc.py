import h5py
from tqdm import tqdm

#sfm_filtered='/outputs/pairs-sfm-filtered.txt'
#pair_path='/outputs/pairs-sfm.txt'
#matches_file='/outputs/matches.h5'

def main(sfm_filtered, pair_path, matches_file):
    with open(sfm_filtered) as filtered_f, open(pair_path, 'w') as orig_f,  h5py.File(matches_file, 'r+') as matches_f:
        total = 0
        filtered = 0
        for l in tqdm(filtered_f):
            total += 1
            i1, i2, res = l.split()
            res =int(res)
            if not res:
                del matches_f[i1][i2]
                filtered += 1
            else:
                orig_f.write(f"{i1} {i2} \n")
        print(f"Doppelgangers Filtered {filtered / total * 100 :.2f}%")