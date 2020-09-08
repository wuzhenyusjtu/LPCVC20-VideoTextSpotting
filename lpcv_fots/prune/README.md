## Prune and Finetune Model

Install [NNI](https://github.com/microsoft/nni) as suggested on official website. Use ```prune_fots.py``` to prune and finetune the model. Get masked pruned model (pruned channels with zero-weight). 

```sh
python3 prune_fots.py --train-folder-syn /path/to/SynthText --train-folder-sample /path/to/SampleDataset\
--batch-size 64 --batches-before-train 2 --ngpus 1 --save-dir /path/to/save/dir --epoch 30 --val \
--pretrain-model /path/to/pretrained/model
```

## Export Pruned Model

Use ```export_pruned_model.py``` to export actual pruned model from pruned masked model. After we export the model, we finetune it with Merged Dataset.

```sh
 python3 export_pruned_model.py --save-dir /path/to/save/dir --prune-model /path/to/nni/pruned/model
```
