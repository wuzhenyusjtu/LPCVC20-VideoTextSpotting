# [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671) text detection branch reimplementation ([PyTorch](https://pytorch.org/))

Note: Original code borrow from https://github.com/Wovchena/text-detection-fots.pytorch

## LPCV Fots Detection Part Pipeline

We did the following steps in our work:
1. Train Fots detection model from scratch with Merged Dataset. Check [standard](./standard)
2. Prune Fots model with [NNI](https://github.com/microsoft/nni). Check [prune](./prune)
3. Prune out channels with zero-weight and get pruned model. Check [prune](./prune)
4. Finetune pruned model with Merged Dataset. Check [standard](./standard)
5. Add rejector to Fts model, freeze other parts and only train rejector. Check [rejector](./rejector)
5. Quantize Fots model with pytorch quantization package. Check [quantize](./quantize)

## Train for LPCV
### Dataset

(1) Merged Dataset

In LPCV video track challenge, we merge part of SynthText Dataset with our own sample dataset which we got from all videos given by the organizer. We make the annotations format of the sample dataset the same with SynthText Dataset. Two paths should be passed to specify paths to two different datasets. <br>
Each epoch randomly we select about 15000 images from first 80% SynthText dataset and all images from sample training dataset for training.
If set val as true, each epoch randomly select about 3500 images from last 20% SynthText dataset and all images from sample test dataset for validation.

(2) Only SynthText Dataset

You can also choose to only use SynthText Dataset to train the model. The path to SynthText dataset should be passed. 

(3) Your own dataset

If you want to use your own dataset, please check ```datasets.py``` for more information. Also, you can find data preprocessing methods in ```datasets.py```. 

### Train model

We train the fots twice in our pipeline. We first train Fots model from scratch in step 1 mentioned above. Then we finetune the pruned model for the second time in step 4. 

### Train original model
Original model can be found in ```model.py```. 
1. Train from scratch with Merged dataset 
    ```sh
    python3 train_sample.py --train-folder-syn /path/to/SynthText --train-folder-sample /path/to/SampleDataset \
    --batch-size 64 --batches-before-train 2 --ngpus 1 --save-dir /path/to/save/dir --epoch 30 --val
    ```
2. Train with pretrained model with Merged dataset 
    ```sh
    python3 train_sample.py --train-folder-syn /path/to/SynthText --train-folder-sample /path/to/SampleDataset \
    --batch-size 64 --batches-before-train 2 --ngpus 1 --save-dir /path/to/save/dir --epoch 30 --val \
    --pretrain-model /path/to/pretrained/model --continue-training
    ```
3. Train with pretrained model with only SynthText dataset 
    ```sh
    python3 train_sample.py --train-folder-syn /path/to/SynthText --batch-size 64 --batches-before-train 2 \
    --ngpus 1 --save-dir /path/to/save/dir --epoch 30 --val \
    --pretrain-model /path/to/pretrained/model --continue-training
    ```
4. Parameter explanation:

    ```--train-folder-syn``` path to the synthtext dataset<br>
    ```--train-folder-sample``` path to the sample dataset<br>
    ```--batch-size``` set batch size for training<br>
    ```--batches-before-train``` how many steps taken before backforward, set 2 as default<br>
    ```--ngpus``` set number of gpus to use<br>
    ```--save-dir``` path you want to save your models<br>
    ```--epochs``` how many epochs to train<br>
    ```--val``` If set, will use validation set to test performance each epoch after training. <br>

### Train pruned model

After we got the pruned model, we finetuned the model with Merged Dataset. Pruned model can be found in ```model_pruned.py```. 

1. Train pruned model with Merged dataset 
    ```sh
    python3 train_sample.py --train-folder-syn /path/to/SynthText --train-folder-sample /path/to/SampleDataset \
    --batch-size 64 --batches-before-train 2 --ngpus 1 --save-dir /path/to/save/dir --epoch 30 --val \
    --pretrain-model /path/to/pruned/model --continue-training --prune 
    ```
2. Parameter explanation

    ```--prune``` If set, will train pruned model. The pretrain model given should be pruned one. <br>

    
### Test/Validate model

You can test the model's performance by running test_sample.py file. Only supports sample dataset. More details please check the code.

1. Test original trained model

```sh
    python3 test_sample.py --test-folder-sample /path/to/SampleDataset --batch-size 1 --batches-before-train 1 --pretrain-model /path/to/pretrained/model
```

2. Test exported pruned model

```sh
    python3 test_sample.py --test-folder-sample /path/to/SampleDataset --batch-size 1 --batches-before-train 1 --pretrain-model /path/to/pruned/model --prune
```

### Model Link

[Pretrained original model](https://drive.google.com/file/d/1LTlveonAFBthWphUYNOJDXJE5SCFOhkD/view?usp=sharing), [Pruned pretrained model](https://drive.google.com/file/d/1SFwTCUBmjOrxNpwR8SyGeLeOyEgdLbPu/view?usp=sharing)

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

## Train Rejector

After we finetune the pruned model, we train rejector for early exit. Our thought is that it is much easier to answer whether there is text in one frame comparing to answer where is the text. We add two conv layers and one fully connected layer after block 2 of Resnet as rejector. Details can be found in ```train_rejector.py``` and ```model_q.py```

1. Dataset 

    We run the exported pruned model on the Sample Dataset. If one image's predicted result is None, we set the label for this image as 0, otherwise set its label as. More information please check ```train_rejector```.

2. Train rejector

    We freeze the original weights and only train the rejector. Run the command:

    ```sh
    python3 train_rejector.py --rejector-folder /path/to/rejector/dataset --save-dir /path/to/save/dir \
    --epochs 6 --pretrain-model /path/to/export/pruned/model
    ```

## Quantize Model

After we finetuned, pruned fots model, the final step is to quantize it. For quantization details, please check ```quantize_model.py```. Run the command:

```sh
python3 quantize_model.py --calibrate-folder /path/to/calibrate/dir --pretrain-model /path/to/pretrain/model \
--backends fbgemm --save-dir /path/to/save/dir
```

