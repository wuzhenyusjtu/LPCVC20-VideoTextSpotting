## Installation
```
conda create --name fots_e2e python=3.6
conda activate fots_e2e
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install opencv-python nltk
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone --branch rahul https://github.com/wuzhenyusjtu/CVPR20-LPCVC.git
cd FOTS_text_spotting
python setup.py build develop
```

## Dataset
- Sample Dataset [[dataset]]()
- Totaltext training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/3eq5ti7z45qfq5gu96gg5t1xwh1yrrt7.zip) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[code]](https://github.com/cs-chan/Total-Text-Dataset). 
- MLT [[dataset]](https://universityofadelaide.box.com/s/tsiimvp65tkf7dw1nuh8l71cjcs0fyif) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/alta996w4fym6arh977h3k3xv55clhg3) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/7k7d6nvf951s4i01szs4udpu2yv5dlqe)

## Quick Start 

### Inference and Evaluation for Trained Models

1. Run inference_e2e with the below command

```
python inference_e2e.py --config-file configs/conf_e2e_finetune.yaml --opts MODEL.WEIGHTS trained_models/finetune/model_final.pth
```

### Train Your Own Models

Pretrainining with synthetic data:

```
OMP_NUM_THREADS=1 python train_recognizer.py \
    --config-file configs/conf_e2e_pretrain.yaml \
    --num-gpus 8 \
    OUTPUT_DIR trained_models/pretrain
```

Finetuning on Sample Dataset:

```
OMP_NUM_THREADS=1 python train_recognizer.py \
    --config-file configs/conf_e2e_finetune.yaml \
    --num-gpus 8 \
    MODEL.WEIGHTS trained_models/pretrain/model_final.pth
```
