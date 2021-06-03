from shutil import copyfile
import os
from tqdm import tqdm
if __name__ == '__main__':
    N = 20000
    src_path = 'data/CelebA/img_align_celeba/img_align_celeba'
    tgt_root = 'data/CelebA/copy'
    for i in tqdm(range(350, N)):
        tgt_dir = 'part{}'.format(i//100)
        tgt_path = os.path.join(tgt_root, tgt_dir)
        if i % 100 == 0:
            os.makedirs(tgt_path, exist_ok=True)

        im_n = "{:06d}.jpg".format(i+1)

        copyfile(os.path.join(src_path, im_n),
                 os.path.join(tgt_path, im_n))
