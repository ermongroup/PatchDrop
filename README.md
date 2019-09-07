# PatchDrop - Learning When and Where to Sample Conditioned on the Low Quality Data

PatchDrop proposes a reinforcement learning setting to perform conditional dynamic image sampling for the image recognition task. Below, you can find the instructions for training it. Our implementation uses Python2.7 and PyTorch framework.

To be able to train on the **fMoW** dataset, you need to download images. You can find the instructions [here](https://github.com/fMoW/dataset). Then, you need to crop the images from the large satellite images based on the bounding boxes provided in the '.json' files. The original fMoW paper adaptively determine the context and add it to the bounding box to find the final area of interest. We follow their strategy to preprocess the images. After processing images, you need to create a **csv** file with two columns:(1) label, (2) location. Label represents the class ID of the image and Location represents the location of the corresponding image. You need to create another .csv file for the validation and test sets. After creating the csv files, transfer them to the directory __./data/fMoW/train.csv__ and __./data/fMoW/test.csv__.

On the other hand, our implementation uses TorchAPI to download CIFAR10, CIFAR100 images. To run it on ImageNet, you need to follow the guidelines [here](https://github.com/soumith/imagenet-multiGPU.torch#data-processing).

## Train the High and Low Resolution Classifiers
In the first step, the high and low resolution classifiers need to be trained on high or low resolution images. To do so, please use the following commands.

How to Run on the **CIFAR10** and **CIFAR100** Datasets:

    python classifer_training.py
       --model R32_C10, R32_C100
       --lr 1e-1
       --cv_dir checkpoint directory
       --batch_size 64
       --penalty -0.5

How to Run on the **fMoW** Dataset(uses ImageNet pretrained model):

    python classifier_training.py
       --model R34_fMoW
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 64
       --penalty -0.5

## Pretrain the Policy Network - Agent
In the second step, the policy network is trained using our reinforcement learning setting and high resolution classifier's predictions. To do so, please use the following commands.

How to Run on the **CIFAR10** and **CIFAR100** Datasets:

    python pretrain.py
       --model R32_C10, R32_C100
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 512
       --penalty -0.5

How to Run on the fMoW Dataset:

    python pretrain.py
       --model R34_fMoW
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 1024
       --penalty -0.5

## Finetune the Policy Network and High Resolution Classifier
In this step, we finetune the Policy Network and High Resolution Classifier jointly. To do so, please use the following command.

How to Run on the **CIFAR10** and **CIFAR100** Datasets:

    python finetune.py
       --model R32_C10, R32_C100
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --Load Load from the latest checkpoint (agent)
       --ckpt_hr_cl Load from the latest checkpoint (hr_classifier)
       --penalty -5

How to Run on the fMoW Dataset:

    python finetune.py
       --model R34_fMoW
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --Load Load from the latest checkpoint (agent)
       --ckpt_lr_cl Load from the latest checkpoint (lr_classifier)
       --penalty -7

## Finetune the Policy Network using Two Stream Classifier (Optional)
This step helps the policy network to drop further patches given the existence of low resolution classifier.

How to Run on the **CIFAR10** and **CIFAR100** Datasets:

    python finetune2stream.py
       --model R32_C10, R32_C100
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --load Load from the latest checkpoint (agent+hr_classifier)
       --ckpt_lr_cl Load from the latest checkpoint (lr_classifier)
       --penalty -5

How to Run on the fMoW Dataset:

    python finetune2stream.py
       --model R34_fMoW
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --load Load from the latest checkpoint (agent+hr_classifier)
       --ckpt_lr_cl Load from the latest checkpoint (lr_classifier)
       --penalty -20
