Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run demo
--------
A demo program can be found in ``demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![Example Image](./data/demo.png)

Expected output:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Train a model from with trained model loaded for LPCV
-----------------
Command:
```sh
python3 train_mj.py --adadelta --trainRoot /share/group_jiliu/yunhexue/MJData/train_new.json  \
--valRoot /share/group_jiliu/yunhexue/MJData/test_new.json --cuda --pretrained ./data/crnn.pth
```
Note:
The training and test dataset are all part of the original dataset. We did not get the pretrained model by training from scratch. We used the model the original repo provided and finetuned it using our own dataset. Here the training and test dataset are all pretrained dataset. We store training and test images paths in .json file. And the model in ```/data``` is also the one given by the original repo. More information can be found in ```dataset.py```

Prune model 
-----------------
Load pretrained model, prune it with [NNI](https://github.com/microsoft/nni) and finetune the pruned model. Run the command:

```sh
python3 prune_model.py --save-dir /path/to/save/dir --pretrained /path/to/pretrained/model (./data/crnn.pth for example) 
```

Export pruned model
-----------------

Export pruned model by running command: (More information please check export_prune_mode.py)

```sh
python3 export_prune_mode.py  --pretrained /path/to/pruned/model --expr_dir /path/to/save/dir
```

Quantize pruned model
-----------------

Please check ```quantize_model.py``` for detailed information.

Train a new model 
-----------------
1. Construct dataset following [origin guide](https://github.com/bgshih/crnn#train-a-new-model). If you want to train with variable length images (keep the origin ratio for example), please modify the `tool/create_dataset.py` and sort the image according to the text length.
2. Execute ``python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda``. Explore ``train.py`` for details.