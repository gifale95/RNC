# In silico discovery of representational relationships across visual cortex

Here we provide the code from the paper:</br>
"[In silico discovery of representational relationships across visual cortex][paper_doi]".</br>
Alessandro T. Gifford, Maya A. Jastrzębowska, Johannes J.D. Singer, Radoslaw M. Cichy



## 📄 Paper abstract

Seeing is underpinned by a complex interconnected network of multiple brain regions of interest (ROIs) jointly representing visual information. However, the representational content of each ROI is typically studied in isolation, and using limited sets of experimenter-picked stimuli. Here, we addressed this by developing [Relational Neural Control (RNC)][rnc_website]. RNC generates and explores in silico functional magnetic resonance imaging (fMRI) responses for large amounts of images, finding controlling images that align or disentangle responses across ROIs, thus indicating their shared or unique representational content.



## 🌓 RNC variants

We developed two RNC variants:

* **Univariate RNC** controls _univariate_ fMRI responses (i.e., responses averaged over all voxels within an ROI), thus exploring representational relationships for visual information encoded in the strongest activation trends common across all ROI voxels.

* **Multivariate RNC** controls _multivariate_ fMRI responses (i.e., population response of all voxels within a ROI), thus exploring representational relationships for visual information encoded in the multi-voxel response patterns.



## 🚀 RNC tutorials

We created interactive tutorials where you can implement univariate and multivariate RNC on in silico fMRI responses of areas spanning the entire visual cortex for ~150,000 naturalistic images: 73,000 images from the [Natural Scenes Dataset][nsd] ([Allen et al., 2022][allen]); 50,000 images from the [ImageNet 2012 Challenge][imagenet] ([Russakovsky et al., 2015][russakovsky]); 26,107 images from the [THINGS Database][things] ([Hebart et al., 2019][hebart]).

These tutorials are available on either _Google Colab_ ([univariate RNC][uni_rnc_colab], [multivariate RNC][multi_rnc_colab]) or _Jupyter Notebook_ ([univariate RNC][uni_rnc_jupyter], [multivariate RNC][multi_rnc_jupyter]).



## ♻️ Reproducibility

### ⚙️ Installation

This repository contains code to reproduce all paper's results.

To run the code, you first need to install the libraries in the [requirements.txt][requirements] file within an Anaconda environment. Here, we guide you through the installation steps.

First, create an [Anaconda][anaconda] environment with the correct Python version:

```shell
conda create -n rnc_env python=3.9
```

Next, download the [requirements.txt][requirements] file, navigate with your terminal to the download directory, and activate the Anaconda environment previously created with:

```shell
source activate rnc_env
```

Now you can install the libraries with:

```shell
pip install -r requirements.txt
```

Finally, you also need to install the [NEST Python package (version 0.3.7)][nest_git] with:

```shell
pip install -U git+https://github.com/gifale95/NEST.git@0.3.7
```


### 🧰 Data

To run the code you will need to download the following:

* The Neural Encoding Simulation Toolkit (https://github.com/gifale95/NEST).

* The 73,000 stimulus images from the Natural Scenes Dataset (https://naturalscenesdataset.org/).

* The 50,000 ILSVRC-2012 validation images (https://www.image-net.org/challenges/LSVRC/2012/index.php).

* The 26,107 images from THINGS (https://osf.io/jum2f/).

* The in vivo fMRI responses for the controlling images collected in this project (https://openneuro.org/datasets/ds005503).

* The Visual Illusion Reconstruction Dataset (https://figshare.com/articles/dataset/Reconstructing_visual_illusory_experiences_from_human_brain_activity/23590302).



### 📦 Code description

* **[`00_generate_insilico_fmri_responses`](https://github.com/gifale95/RNC/tree/main/00_generate_insilico_fmri_responses):** Generate the in silico fMRI responses later used by RNC.
* **[`01_in_silico_fmri_encoding_accuracy`](https://github.com/gifale95/RNC/tree/main/01_in_silico_fmri_encoding_accuracy):** Compute the encoding model's prediciton accuracy, and perform a noise analysis on the in silico fMRI responses.
* **[`02_univariate_rnc`](https://github.com/gifale95/RNC/tree/main/02_univariate_rnc):** Apply univariate RNC on the in silico fMRI responses.
* **[`03_generative_univariate_rnc`](https://github.com/gifale95/RNC/tree/main/03_generative_univariate_rnc):** Apply generative univariate RNC on the in silico fMRI responses.
* **[`04_multivariate_rnc`](https://github.com/gifale95/RNC/tree/main/04_multivariate_rnc):** Apply multivariate RNC on the in silico fMRI responses.
* **[`05_multivariate_rnc_retinotopy`](https://github.com/gifale95/RNC/tree/main/05_multivariate_rnc_retinotopy):** Perform the retinotopy analysis on the in silico fMRI resposnes for the V1 vs. V4 multivariate RNC controlling images.
* **[`06_rnc_categorical_slectivity`](https://github.com/gifale95/RNC/tree/main/06_rnc_categorical_slectivity):** Apply the univariate and multivariate RNC categorical selectivity analysis on in silico fMRI responses for high-level visual areas.
* **[`07_multidimensional_scaling`](https://github.com/gifale95/RNC/tree/main/07_multidimensional_scaling):** Apply multidimensional scaling on the in silico fMRI responses for RNC's controlling images.
* **[`08_in_vivo_validation`](https://github.com/gifale95/RNC/tree/main/08_in_vivo_validation):** Analyze the in vivo fMRI responses for the V1 vs. V4 univariate and multivariate RNC controlling images.



## ❗ Issues

If you experience problems with the code, please get in touch with Ale (alessandro.gifford@gmail.com), or submit an issue.



## 📜 Citation
If you use any of our data or code, please cite:

> * Gifford AT, Jastrzębowska M, Singer JJD, Cichy RM. 2024. In silico discovery of representational relationships across visual cortex. _arXiv preprint_, arXiv:2411.10872. DOI: [https://doi.org/10.48550/arXiv.2411.10872][paper_doi]
> * Gifford AT, Bersch D, Roig G, Cichy RM. 2025. The Neural Encoding Simulation Toolkit. _In preparation_. https://github.com/gifale95/NEST



[paper_doi]: https://doi.org/10.48550/arXiv.2411.10872
[rnc_website]: https://www.alegifford.com/projects/rnc/
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[nest_website]: https://www.alegifford.com/projects/nest/
[imagenet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
[russakovsky]: https://link.springer.com/article/10.1007/s11263-015-0816-y
[things]: https://things-initiative.org/
[hebart]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792
[uni_rnc_colab]: https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing
[multi_rnc_colab]: https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing
[uni_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/univariate_rnc_tutorial.ipynb
[multi_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/multivariate_rnc_tutorial.ipynb
[requirements]: https://github.com/gifale95/RNC/blob/main/requirements.txt
[anaconda]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
[nest_git]: https://github.com/gifale95/NEST
