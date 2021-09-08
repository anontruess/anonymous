This repository is strictly meant for reviewers.
Please follow the instructions below to reproduce the paper.

## Dependencies and Installation
We strongly recommend to use Nvidia-Docker for reproducibility. Install both `docker` and `nvidia-docker` (please find instructions from their respective installation pages).
After the docker installations, pull pytorch docker image with the following command:
`docker pull nvcr.io/nvidia/pytorch:18.04-py3`
and run the image using the command:
`nvidia-docker run --rm -ti --ipc=host nvcr.io/nvidia/pytorch:18.04-py3`

Install software requirements as listed in [requirements.txt](requirements.txt) by typing, `pip install -r requirements.txt`

The code has been run and tested on `Python 3.6.4`, `Cuda 9.0, V9.0.176` and `PyTorch 0.4.1`. 

## Dataset

If running the Distribution Matching stage, downloading dataset isn't required. The extracted features can be used directly from release binaries or from [here](https://yadi.sk/d/HS9Sj4k5IRmc-g?w=1). Else, place the dataset in `../dataset/` folder or just switch `download=True` in [simclr_main.py](simclr_main.py).

# Usage

## Pretrained-Models

It is recommended to run the Distribution Matching stage directly using the extracted features from release binaries or from [here](https://yadi.sk/d/HS9Sj4k5IRmc-g?w=1).
The pretrained models can be downloaded from release binaries.


## Running Distribution-Matching

* Make sure the features are extracted and present in `models_stage_simclr/train2/class_activation_{train or test}/`. 
* Create the `log-path` directory.
* Then, run `python sinkhorn_main.py --log_path log`.

## Running the complete pipeline

* For Stage 1 Backbone training, run `python simclr_main.py`.
* Run `python sinkhorn_activations.py --layer 6 --split {train or test}` to dump the features or activations.
* For Stage 2 Feature Clustering (FC), run `python sinkhorn_clusters.py`.
* For Stage 3 Distribution Matching, run `python sinkhon_main.py --log_path log`.
