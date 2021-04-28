# Minimal script for generating images using pre-trained the GANsformer
# Ignore all future warnings
from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm

from training import misc
from training.misc import crop_max_rectangle as crop

import dnnlib.tflib as tflib
from pretrained_networks import load_networks # returns G, D, Gs
# G: generator, D: discriminator, Gs: generator moving-average (higher quality images)


def shuffle1(data):
    for i in range(0,data.shape[0],2):
        backup = np.copy(data[i])
        data[i][0::2] = data[i + 1][0::2]
        data[i + 1][0::2] = backup[0::2]

def shuffle2(data):
    data1 = np.copy(data[0])
    data2 = np.copy(data[1])
    merge = np.concatenate([data1[data1.shape[0] // 2:,:],data2[:data2.shape[0] // 2,:]])
    merge = np.expand_dims(merge,axis=0)

    return merge

def run(model, gpus, output_dir, images_num, truncation_psi, batch_size, ratio):
    print("Loading networks...")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus                   # Set GPUs
    tflib.init_tf()                                             # Initialize TensorFlow
    G, D, Gs = load_networks(model)                             # Load pre-trained network

    print("Generate images...")
    with open('./latents.pkl', 'rb') as f:
        latents = pickle.load(f)
    
    idx_source = 4
    idx_ref = 5
    
    # for each latent code
    for i in range(latents.shape[1]):
        
        buff = np.copy(latents[idx_ref])

        buff[:i,:] = 0
        buff[i+1:,:] = 0
        target = latents[idx_source] + buff

        latents = np.linspace(latents[idx_source], target, 32)

        images = Gs.run(latents, truncation_psi = truncation_psi,   # Generate images
            minibatch_size = batch_size, verbose = True)[0]

        print("Saving images...")
        sub_dir = "/latent{:02d}".format(i)
        target_dir = output_dir + sub_dir
        os.makedirs(target_dir, exist_ok = True)                    # Make output directory
        for j, image in tqdm(list(enumerate(images))):              # Save images
            filepath = "{}/Sample_{:03d}.png".format(target_dir,j)       # Output images pattern
            croped = crop(misc.to_pil(image), ratio)
            croped.save(filepath)

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANsformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume (optional)", default = None, type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--batch-size",         help = "Batch size for generating images (default: %(default)s)", default = 1, type = int)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    args = parser.parse_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
