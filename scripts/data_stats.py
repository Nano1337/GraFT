from multiprocessing import Pool
import os
import numpy as np
from PIL import Image


def calc_stats(filename):
    im = Image.open(filename)
    im_arr = np.array(im)
    im_arr = im_arr / 255.0

    # return mean and std for each channel 
    return (im_arr.mean(axis=(0, 1)), im_arr.std(axis=(0, 1)))


if __name__ == '__main__':
    root_folder = r"/data/datasets/research-datasets/reid_mm/RGBNT100/train/T"
    # crawl through all subfolders of root_folder and append all filenames that end in .png to a list
    filenames = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                filenames.append(os.path.join(root, file))

    num_processes = 16

    with Pool(num_processes) as p:
        results = p.map(calc_stats, filenames)

    # show mean and std for each channel
    means = [r[0] for r in results]
    stds = [r[1] for r in results]
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")
