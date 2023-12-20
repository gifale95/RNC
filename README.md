# Disentangling the unique contribution of human retinotopic regions using neural control

Here we provide the code to reproduce the results of our data resource paper:</br>
"[Disentangling the unique contribution of human retinotopic regions using neural control][paper_link]".</br> --> Edit paper title !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Alessandro T. Gifford, Johannes J.D. Singer, Maya Jastrzebowska, Radoslaw M. Cichy

If you experience problems with the code, please create a pull request or report the bug directly to Ale via email (alessandro.gifford@gmail.com).



## Environment setup --> Create environment file, and add right links to it  !!!!!!!!!!!!!!!!!!!!!!!
To run the code first install [Anaconda][conda], then create and activate a dedicated Conda environment by typing the following into your terminal:
```shell
curl -O https://raw.githubusercontent.com/gifale95/neural_control/main/environment.yml
conda env create -f environment.yml
conda activate neural_control
```
Alternatively, after installing Anaconda you can download the [environment.yml][env_file] file, open the terminal in the download directory and type:
```shell
conda env create -f environment.yml
conda activate neural_control
```


## Data availability !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
The Natural Scenes Dataset (NSD) is available at [www.naturalscenesdataset.org][nsd]. The THINGS EEG2 dataset is available on [OSF][things_eeg_2]. The neural control stimuli, along with their collected fMRI and EEG responses, are available on [OSF][new_collected_data]. To run the code, you need to download the data and place it into the following directories:

* **NSD:** `../project_directory/???`
* **THINGS EEG2:** `../project_directory/???`
* **Neural control stimuli:** `../project_directory/???`
* **Collected fMRI responses:** `../project_directory/???`.
* **Collected EEG responses:** `../project_directory/???`.



## Code description !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* **01_???:** ??? description ???.



## Cite !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
If you use any of our data or code, partly or as it is, please cite our paper:

Gifford AT, Singer JJD, Jastrzebowska M, Cichy RM. 2023. Disentangling the unique contribution of human retinotopic regions using neural control. _bioRxiv_, ???. DOI: [???][paper_link]



[paper_link]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[conda]: https://www.anaconda.com/
[env_file]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[nsd]: https://www.naturalscenesdataset.org/
[things_eeg_2]: https://osf.io/3jk45/
[new_collected_data]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
