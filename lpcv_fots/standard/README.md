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
