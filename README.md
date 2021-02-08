# RythmGAN_pytorch
The pytorch implementation for Conditional RythmGAN+TransformerKernal

![alt text](https://github.com/MCLYang/RhythmGAN_pytorch/blob/master/img/Screenshot%20from%202021-02-08%2002-33-00.png)

This project is the pytorch implemtation for Conditional Generative Adversarial Network(CGAN) to generate the rythme. Please cite the original work from Nao Tokui for more detail: https://cclab.sfc.keio.ac.jp/projects/rhythmcan/

The Keras implemntation:https://github.com/naotokui/CreativeGAN-Rhythm

Conditional Generative Adversarial Nets: https://arxiv.org/abs/1411.1784

Besides the LSTM kernal, we are inspired by [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and provide Transformer kernal via transformer encoder block.

## Usage
Option 1: Train on Google Colab
- Train RythmGAN using LSTM Kernal https://colab.research.google.com/drive/1JOVz0n1jz-rSIkEDnCEm7omqATAwpuk5?usp=sharing

- Train RythmGAN using Transformer Kernal https://colab.research.google.com/drive/1JOVz0n1jz-rSIkEDnCEm7omqATAwpuk5?usp=sharing
> download dataset

`wget https://www.dropbox.com/s/a8mwk8rdv08cu2l/data.zip`

`unzip data.zip`
