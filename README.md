# In silico discovery of representational relationships across visual cortex

Here we provide the code from the paper:</br>
"[In silico discovery of representational relationships across visual cortex][paper_doi]".</br>
Alessandro T. Gifford, Maya A. Jastrzębowska, Johannes J.D. Singer, Radoslaw M. Cichy



## 📖 Theoretical motivation

Seeing is underpinned by a complex interconnected network of multiple brain regions of interest (ROIs) jointly representing visual information. However, the representational content of each ROI is typically studied in isolation, and using limited sets of experimenter-picked stimuli. Here, we addressed this by developing [Relational Neural Control (RNC)][rnc_website]. RNC generates and explores in silico functional magnetic resonance imaging (fMRI) responses for large amounts of images, finding controlling images that align or disentangle responses across ROIs, thus indicating their shared or unique representational content.

We developed two RNC variants:

* **Univariate RNC** controls _univariate_ fMRI responses (i.e., responses averaged over all voxels within an ROI), thus exploring representational relationships for visual information encoded in the strongest activation trends common across all ROI voxels.
* **Multivariate RNC** controls _multivariate_ fMRI responses (i.e., population response of all voxels within a ROI), thus exploring representational relationships for visual information encoded in the multi-voxel response patterns.

To thoroughly explore the visual space in search for controlling stimuli, we applied RNC to in silico neural responses for thousands of naturalistic images, that is, neural responses generated through encoding models from the [Neural Encoding Dataset (NED)][ned_website]. Specifically, this includes NED-generated in silico fMRI responses for the 73,000 [Natural Scenes Dataset][nsd] ([Allen et al., 2022][allen]) images, the 50,000 [ImageNet 2012 Challenge][imagenet] ([Russakovsky et al., 2015][russakovsky]) images, or the 26,107 [THINGS Database][things] ([Hebart et al., 2019][hebart]) images.

For any question regarding this code, the project data, or RNC in general, you can get in touch with Ale (alessandro.gifford@gmail.com).



## 🚀 RNC tutorials

We created interactive tutorials where you can learn how to use univariate and and multivariate RNC. These tutorials are available on either _Google Colab_ ([univariate RNC][uni_rnc_colab], [multivariate RNC][multi_rnc_colab]) or _Jupyter Notebook_ ([univariate RNC][uni_rnc_jupyter], [multivariate RNC][multi_rnc_jupyter]).



## ♻️ Reproducibility

### ⚙️ Installation

To reproduce the paper's results, you can download and run the Python code from this repository. To run this code, you will first need to install the libraries in the [requirements.txt][requiremenents]. We recommend installing these libraries within a virtual environment (e.g., an [Anaconda][anaconda] environment) using:

```shell
pip install -r requirements.txt
```

You will also need to manually install the [NED][ned_git] library with:

```shell
pip install -U git+https://github.com/gifale95/NED.git
```


### 🧰 Data

To run the code you will need to download the following:

* The Neural Encoding dataset (https://www.alegifford.com/projects/ned/).

* The 73,000 stimulus images from the Natural Scenes Dataset (https://naturalscenesdataset.org/).

* The 50,000 ILSVRC-2012 validation images (https://www.image-net.org/challenges/LSVRC/2012/index.php).

* The 26,107 images from THINGS (https://osf.io/jum2f/).

* The in vivo fMRI responses for the controlling images collected in this project (https://openneuro.org/datasets/ds005503).



### 📦 Code description

* **00_generate_insilico_fmri_responses:** Generate in silico fMRI responses for naturalistic images, using trained encoding models from the Neural Encoding Dataset.
* **01_in_silico_fmri_encoding_accuracy:** Compute the encoding accuracy and perform a noise analysis on the in silico fMRI responses.
* **02_univariate_rnc:** Apply univariate RNC on the in silico fMRI responses.
* **03_generative_univariate_rnc:** Apply generative univariate RNC on  in silico fMRI responses.
* **04_multivariate_rnc:** Apply multivariate RNC on the in silico fMRI responses.
* **05_multivariate_rnc_retinotopy:** Perform the retinotopy analysis on the in silico fMRI resposnes for the V1 vs. V4 multivariate RNC controlling images.
* **06_in_vivo_validation:** Analyze the in vivo fMRI responses for the V1 vs. V4 univariate and multivariate RNC controlling images.



## ❗ Issues

If you experience problems with the code, please submit an issue!



## 📜 Citation
If you use any of our data or code, please cite:

> * Gifford AT, Jastrzębowska M, Singer JJD, Cichy RM. 2024. In silico discovery of representational relationships across visual cortex. _arXiv preprint_, arXiv:2411.10872. DOI: [https://doi.org/10.48550/arXiv.2411.10872][paper_doi]
> * Gifford AT, Cichy RM. 2024. The Neural Encoding Dataset. _In preparation_. https://github.com/gifale95/NED



[paper_doi]: https://doi.org/10.48550/arXiv.2411.10872
[rnc_website]: https://www.alegifford.com/projects/rnc/
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[ned_website]: https://www.alegifford.com/projects/ned/
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
[ned_git]: https://github.com/gifale95/NED
