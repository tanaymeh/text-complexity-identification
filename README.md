<h1 align='center'>Text Complexity Identification 📚</h1>

<p align="center">
<img src="assets/image.jpg" alt="Picture for Representation">
</p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

<img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />
</p>

## Introduction

This repository houses the code I wrote for detecting the ease of readability of a given text excerpt which is the main task of the currently running [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/) Competition

The main task in this competition is to take a text excerpt and predict it's ease of readability score (which is a continuous value).

I have also made a [full training notebook](https://www.kaggle.com/heyytanay/training-kfolds-pytorch-bert-large-w-o-oom) in this competition.

## Data

The main data consists of 2 important features (or columns): `excerpt` and `target`.

The `excerpt` is a basically a text sentence with a corresponding `target` value which denotes how "easy it is to read a sentence"

The data consist of 3 files: `train.csv`, `test.csv` and `sample_submission.csv`.

## Training the Model

If you want to train the model on this data as-is, then you would typically have to perform 2 steps:

### 1. Getting the Data right

First, download the data from [here](https://www.kaggle.com/c/commonlitreadabilityprize/data). 

Now, take the downloaded `.zip` file and extract it into a new folder: `input/`.

Make sure the `input/` folder is at the same directory level as the `train.py` file.


### 2. Installing the dependencies

To run the code in this repository, you need a lot of frameworks installed on your system.

Make sure you have enough space on your disk and Internet quota before you proceed.

```shell
$ pip install -r requirements.txt
```

### 3. Training the Model

If you have done the above steps right, then just running the `train.py` script should not produce any errors.

To run training, open the terminal and change your working directory to the same level as the `train.py` file.

Now, for training do:

```shell
$ python train.py
```

This should start training in a few seconds and you should see a progress bar.

If you are having problems related to anything, please open an Issue and I will be happy to help!

**I hope you found my work useful! If you did, then please ⭐ this repository!**