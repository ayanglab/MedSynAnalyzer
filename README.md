#  MedSynAnalyzer

This repository provides a comprehensive set of tools for analyzing the quality of synthetic images. It covers three important aspects: fidelity, variety, and privacy. The analysis is based on features extracted from both 2D and 3D images, making it applicable to a wide range of image types. Pre-trained feature extractors will be made available once they are fully trained.

The features extracted can be either discrete features obtained from VQVAE or continuous features from other pre-trained models. For discrete features, we utilize the Hamming distance to measure feature similarity, and for continuous features, we use Euclidean distance. 

## Method description
This analyzer evaluates synthetic images from three perspectives: fidelity, variety, and privacy.

**Fidelity and privacy.**
To assess fidelity, we consider the similarity between real and synthetic images. For each real image, we identify its k nearest neighbors based on image similarity. We then categorize synthetic images into three sets:


![function](https://github.com/XiaodanXing/synthetic_image_analyzer/assets/30890745/eed19acb-1cf7-40d2-a10c-b70ecb46d113)


- Copy set: Contains synthetic images that closely resemble a specific real image. A synthetic image belongs to this set if it is closer to the real image than any other real images.

- Real set: Comprises synthetic images that are realistic but not exact copies of any real image. A synthetic image is assigned to this set if it falls within the kth nearest neighbor range of the corresponding real image.

- Non-real set: Includes synthetic images that are not realistic compared to the corresponding real image. A synthetic image is placed in this set if it is not within the kth nearest neighbor range of the real image.

Privacy preservation score is computed for each synthetic image based on its assignment to these sets. If a synthetic image is part of the copy set for any real image, its privacy protection ability is considered to be 0.

The fidelity is defined by the ratio of realistic synthetic images (synthetic images that falls to the real and copy set) to all synthetic images. 


**Variety.**

To measure the variety, we introduced the JPEG file size of the mean image. The lossless JPEG file size of the group average image was used to measure the inner class variety in the ImageNet dataset. This approach was justified by the authors who presumed that a dataset containing diverse images would result in a blurrier average image, thus reducing the lossless JPEG file size of the mean image. To ensure that the variety score is positively correlated with the true variety, we normalized it to $[0,1]$ across all groups of synthetic images, and then subtracted it from 1 to obtain the final variety score. 

It is worth noting that variety can also be quantified by the standard deviation (STD) of the latent features, which is also implemented in our package. 


## Usage

We have packaged this tool as a python package. You can simply install this using
```
pip install synthetic-image-analyzer
```

### Image Quality Analysis Command:
To perform image quality analysis on synthetic images, use the analyze command:
```
synthetic_image_analyzer --data_dir ./synthetic_images --ref_dir ./real_images --output_dir ./results --resolution 512 --num_channels 3
```

The package also provides an API for programmatic access to its functionality. An example can be found in ![demo.py](./demo.py), which analyzed the ![MM-FID and MM-STD](https://ieeexplore.ieee.org/abstract/document/10077525) proposed in our TMI submission. 

Multi-scale multi-task Fréchet Inception Distance (MM-FID) and multi-scale multi-task Standard Deviation (MM-STD) incorporate different feature extractors under different resolutions to extract more representative features.

## Applications 
With this open-sourced analyzer, we have performed a comprehensive analysis on the quality of synthetic images for our MICCAI 2023 paper. Through intensive experiments in over 100k chest X-ray images, we drew three major conclusions which we can envision that have broad applicability in medical image synthesis and analysis.

![teaser](https://github.com/ayanglab/MedSynAnalyzer/assets/30890745/fe8f4343-6568-40d6-931d-d91681586fb7)


## Citation
If you find this repository useful to you, please cite our papers:

[1] Xing, X., Papanastasiou, G., Walsh, S., & Yang, G. (2023). Less is More: Unsupervised Mask-guided Annotated CT Image Synthesis with Minimum Manual Segmentations. IEEE Transactions on Medical Imaging.

[2] Xing, X., Nan, Y., Felder, F., Walsh, S., & Yang, G. (2023). The Beauty or the Beast: Which Aspect of Synthetic Medical Images Deserves Our Focus?. arXiv preprint arXiv:2305.09789.
