# Hybrid Linear Attention: A Vision Transformer Integrating Selective Sampling Softmax and Multi-Feature Fusion Enhancement

## Getting Started
```bash
conda create -n hlsmvit python=3.10
conda activate hlsmvit
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txt
```
## Training
```bash
torchpack dist-run -np 1 python train_cls_model.py configs/cls/imagenet/b1.yaml  --path .exp/cls/imagenet/m1_r224
```
## Test
```bash
python eval_cls_model.py --model b1-r224 --image_size 224 --weight_url xxx.pt
```
## Model Weights
```
Download pre-trained weights:
- [Baidu Yun](https://pan.baidu.com/s/1-tgQ0Po637VBIV7DiSLE3g?pwd=0523)
```
