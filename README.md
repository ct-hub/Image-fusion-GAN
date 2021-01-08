# Image fusion GAN
Implementation of a GAN-based method for the fusion of visible-infrared images.

## System requirements

**Note:** For the h5 files to download correctly, you need to have GitLFS installed.

## Model summary
The basic structure is the one proposed by [Zhao et al.](https://www.hindawi.com/journals/mpe/2020/3739040/); for several implementation details (e.g. the number of neurons/filters per layer, among others), I referred to the [Pix2Pix model](https://paperswithcode.com/paper/image-to-image-translation-with-conditional) by Isola et al. for the Generator 1. For the Generator 2 and both discriminators, I referred to the [FusionGAN model](https://www.researchgate.net/publication/327393843_FusionGAN_A_generative_adversarial_network_for_infrared_and_visible_image_fusion) by Ma et al.

I also extended the model by modifying the original architecture of the Generator 1 from an encoder-decoder structure to a U-Net one, modified the output layers of the generators to allow for three-channel output images, and added Spectral Normalization and the Two Time-Scale Update Rule (TTUR) for training stability. For more details on these extensions, see the paper here [link to arXiv or journal].
