# Image fusion GAN
Implementation of a GAN-based method for the fusion of visible-infrared images.

## System specifications
This model was developed and tested on Ubuntu 16.04.6 LTS.

It was also tested on two different versions of Tensorflow:
  * Tensorflow 2.3.1 (tested with no GPU support).
  * Tensorflow [pending] (tested with single GPU support).
  
**Note:** For the h5 files to download correctly, it is necessary to have GitLFS installed.
  
### GPU support
The code should run seamlessly if no GPU is detected. It is designed to work also with a single GPU (tested with a Tesla P100 GPU). You should specify in the **main.py** file the ID of the GPU you are using (even if you only have one). The latter is meant to prevent Tensorflow from allocating memory on multiple GPUs on a shared environment.

## Model summary
The basic structure is the one proposed by [Zhao et al.](https://www.hindawi.com/journals/mpe/2020/3739040/); for several implementation details (e.g. the number of neurons/filters per layer, among others), I referred to the [Pix2Pix model](https://paperswithcode.com/paper/image-to-image-translation-with-conditional) by Isola et al. for the *Generator 1*. For the *Generator 2* and both discriminators, I referred to the [FusionGAN model](https://www.researchgate.net/publication/327393843_FusionGAN_A_generative_adversarial_network_for_infrared_and_visible_image_fusion) by Ma et al.

I also extended the model by modifying the original architecture of the *Generator 1* from an encoder-decoder structure to a U-Net one, modified the output layers of the generators to allow for three-channel output images, and added Spectral Normalization and the Two Time-Scale Update Rule (TTUR) for training stability. For more details on these extensions, see the paper here [link to arXiv or journal].
