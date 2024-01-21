# Sample-specific Multi-channel Masks for Visual Reprogramming
This is the implementation of our paper submitted for ICML2024.

The seeds we use in our experiments are 0, 1, 2.

## Installation
        conda create -n reprogram
        conda activate reprogram
        pip install -r requirements.txt

## Training
        python instancewise_vp.py --dataset cifar10 --network resnet18 --seed 0
        python instancewise_vp.py --dataset cifar10 --network ViT_B32 --seed 0

        
