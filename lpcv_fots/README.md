# [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671) text detection branch reimplementation ([PyTorch](https://pytorch.org/))

Note: Original code borrow from https://github.com/Wovchena/text-detection-fots.pytorch

## LPCV Fots Detection Part Pipeline

We did the following steps in our work:
1. Train Fots detection model from scratch with Merged Dataset. Please check [standard](./standard)
2. Prune Fots model with [NNI](https://github.com/microsoft/nni). Please check [prune](./prune)
3. Prune out channels with zero-weight and get pruned model. Please check [prune](./prune)
4. Finetune pruned model with Merged Dataset. Please check [standard](./standard)
5. Add rejector to Fts model, freeze other parts and only train rejector. Please check [rejector](./rejector)
5. Quantize Fots model with pytorch quantization package. Please check [quantize](./quantize)
