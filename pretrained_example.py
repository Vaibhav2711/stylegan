# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def genFace(num):
    # Initialize TensorFlow.
    tflib.init_tf()
    
    # Print network details.
    _G, _D, Gs = pickle.load(open("karras2019stylegan-ffhq-1024x1024.pkl","rb"))
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(num)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    filename = "image_"+str(num)+".png"
    png_filename = os.path.join(config.result_dir, filename)
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    return config.result_dir+"/"+filename


