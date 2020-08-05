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
- Sample Dataset [[link]](https://drive.google.com/file/d/12wAhuYJWizs_kMTd7lYcurnFd7Ntc9Y_/view?usp=sharing)
- Totaltext training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/3eq5ti7z45qfq5gu96gg5t1xwh1yrrt7.zip)
- MLT [[link]](https://universityofadelaide.box.com/s/tsiimvp65tkf7dw1nuh8l71cjcs0fyif)
- Syntext-150k: 
  - Part1: 94,723 [[link]](https://universityofadelaide.box.com/s/alta996w4fym6arh977h3k3xv55clhg3) 
  - Part2: 54,327 [[link]](https://universityofadelaide.box.com/s/7k7d6nvf951s4i01szs4udpu2yv5dlqe)

## Quick Start 

### Inference and Evaluation for Trained Models

1. Run inference_e2e with the below command

```
python inference_e2e.py \
    --config-file configs/conf_e2e_finetune.yaml \
    --opts MODEL.WEIGHTS trained_models/finetune/model_final.pth
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
