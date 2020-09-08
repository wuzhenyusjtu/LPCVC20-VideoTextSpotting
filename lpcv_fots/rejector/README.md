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
