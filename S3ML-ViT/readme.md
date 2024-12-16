# Effective Linear Vision Transformer Via Selective Sampling Softmax and Multi-Feature Enhancement

## Getting Started
```bash
conda install --yes --file requirements.txt
```
## Training
```bash
torchpack dist-run -np 1 python train_cls_model.py configs/cls/imagenet/b1.yaml  --path .exp/cls/imagenet/m1_r224
```
## Test
```bash
python eval_cls_model.py --model b1-r224 --image_size 224 --weight_url xxx.pt
```
