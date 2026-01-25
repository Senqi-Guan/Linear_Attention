## Getting Started

### Installation

```bash
conda create -n accap python=3.10
conda activate accap
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txt
```

## Training
'''
torchpack dist-run -np 4 python train_cls_model.py configs/cls/imagenet/b1.yaml --path /path/to/savepath
'''

## Evaluation
'''
python eval_cls_model.py --model b1-r224 --image_size 224 --weight_url /path/to/weights --path /path/to/imagenet/val
'''
