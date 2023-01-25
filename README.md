# Advancing Radiograph Representation Learning with Masked Record Modeling (MRM)
This is the Official Implement of the paper: [Advancing Radiograph Representation Learning with Masked Record Modeling (ICLR'23)](https://openreview.net/forum?id=w-x7U26GM7j).

Some code of this repository is borrowed from [MAE](https://github.com/facebookresearch/mae), [huggingface](https://huggingface.co) and [REFERS](https://github.com/funnyzhou/REFERS).

## Getting started
### 1 Environmental Requirement
- Ubuntu 18.04 LTS.

- Python 3.8.11

If you are using anaconda/miniconda, we provide an easy way to prepare the environment for pre-training and finetuning of classification:

      conda env create -f environment.yaml
      pip install -r requirements.txt

### 2 Pre-training
#### 2.1 Data preparation for pre-training
- We use MIMIC-CXR-JPG for pre-training. You can acquire more information about this dataset at [Johnson et al. MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
- The dataset directory specified in [run.sh](/run.sh) includes the MIMIC-CXR-JPG dataset and you need to prepare a file "training.csv" and put it into the dataset directory.
- The file "training.csv" includes two columns  "image_path" and "report_content" for each line, corresponding to 1. the path to an image, 2. the text of the corresponding report.

  

#### 2.2 Start pre-training
- Download the pre-trained weights of [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and set "resume" to the path of pre-trained weights in [run.sh](/run.sh).
- Set the data path, GPU IDs, batch size, output directory, and other parameters in [run.sh](/run.sh).

- Start training by running

      chmod a+x run.sh
      ./run.sh

### 3 Fine-tuning of classification (take NIH ChestX-ray 14 dataset as the example)
#### 3.1 Data preparation
- Download NIH ChestX-ray 14 dataset and split [train/valid/test](DatasetsSplits/NIH_ChestX-ray) set. The directory should be organized as follows:

      NIH_ChestX-ray/
            all_classes/
                  xxxx1.png
                  xxxx2.png
                  ...
                  xxxxn.png
            train_1.txt
            trian_10.txt
            train_list.txt
            val_list.txt
            test_list.txt	
- Specify the "dataset_path" in [finetuning_1percent.sh](/NIH_ChestX-ray/finetuning_1percent.sh), [finetuning_10percent.sh](/NIH_ChestX-ray/finetuning_10percent.sh), [finetuning_100percent.sh](/NIH_ChestX-ray/finetuning_100percent.sh), [test.py](/NIH_ChestX-ray/test.py).

#### 3.2 Start fine-tuning (take 1 percent data as the example)
- Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1JwZaqvsSdk1bD3B7fsN0uOz-2Fzz1amc/view?usp=sharing) and specify "pretrained_path" in [finetuning_1percent.sh](/NIH_ChestX-ray/finetuning_1percent.sh).

- Start training by running

      chmod a+x finetuning_1percent.sh
      ./finetuning_1percent.sh

### 4 Fine-tuning of segmentation
#### 4.1 Data preparation
- Download SIIM-ACR Pneumothorax and preprocess the images and annotations.
Then organize the directory as follows:

      siim/
            images/
                  training/
                        xxxx1.png
                        xxxx2.png
                        ...
                        xxxxn.png
                  validation/
                        ...
                  test/
                        ...

            annotations/
                  training/
                        xxxx1.png
                        xxxx2.png
                        ...
                        xxxxn.png
                  validation/
                        ...
                  test/
                        ...

#### 4.2 Necessary files for segmentation
We conduct all experiments  of segmentation by [MMSegmentaiton](https://github.com/open-mmlab/mmsegmentation) (version  0.25.0) and it is necessary to set the environment and comprehend the code structures of MMSegmentaiton in advance.

Here we provide the necessary configuration files for reproducing the experiments  in the directory "Siim _Segmentation".


### 5 Links to download datasets
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

- [NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/#:~:text=What%20is%20CheXpert%3F,labeled%20reference%20standard%20evaluation%20sets.)

- [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

- [COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)

- [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)