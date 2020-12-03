Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run demo
--------
This part is from original CRNN repo.

A demo program can be found in ``demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.


Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Get MJDataset training/testing json file
-----------------
First we create json files for training and testing. ```get_data_json.py``` aims at generating the train json file and test json file. We need to pass the path to [MJ Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/) and set the number of training and testing data. Run ```get_data_json.py``` as follows:
```sh
python3 get_data_json.py --datadir path/to/mj/dataset --train_num number_for_training --test_num number_for_testing
```
For example, we can run it as:
```sh
python3 get_data_json.py --datadir ../mnt/ramdisk/max/90kDICT32px --train_num 3000000 --test_num 30000
```
Then we can find two json files in the same path of the dataset.

Train a model from scratch
-----------------
Command:
```sh
python3 train.py --adadelta --trainRoot  /path/to/train/json/file --valRoot /path/to/test/json/file --cuda 
--expr_dir /path/to/saved/dir
```
Use MJDataset to get the pretrained model. The two paths are the path to the two json files mentioned above. ```expr_dir``` parameter is the path to the saved models.
```adadelta``` is suggested. More parameters can be found in ```train_mj.py```. 

Prune model 
-----------------
Load pretrained model, prune it with [NNI](https://github.com/microsoft/nni) and finetune the pruned model. Run the command:

```sh
python3 train.py --adadelta --trainRoot /path/to/train/json/file --valRoot /path/to/test/json/file --cuda 
--expr_dir /path/to/save/dir --prune --pretrained /path/to/pretrained/model (./data/crnn.pth for example) 
```
```--prune``` parameter is set to do finetuning with ```train.py```. Now we have the pruned model with channels set to 0. To get the real pruned model, we need to prune out all zero-channels and export pruned model.

Export pruned model
-----------------

Export the pruned model by running command: (More information please check export_prune_mode.py)

```sh
python3 export_pruned_crnn.py  --pretrained /path/to/pruned/model --expr_dir /path/to/save/dir
```
Then you have a pruned model with lighter architecture at the target directory.

Finetune the pretrained model with sample dataset
-----------------
After we export the pruned model, we need to finetune our model with sample dataset.
Command:
```sh
python3 train.py --adadelta --trainRoot /path/to/training/sample/crnn/dataset --valRoot /path/to/test/sample/crnn/dataset 
--cuda --expr_dir /path/to/saved/dir --finetune --pretrained /path/to/exported/model
```
The dataset contains images with labels as their names. The dataset can be found in ```/data/yunhe/sample_bezier_all``` on 64.38.150.214 server. More details can check ```dataset.py``` and ```train.py```. ```--finetune``` parameter is set to do finetuning with ```train.py```.

Quantize pruned model
-----------------

Command:
```sh
python3 quantize_model.py --pruned /path/to/exported/model --expr_dir /path/to/saved/dir --qconfig qnnpack
```
```--qconfig``` parameter sets the configure of quantization. It should be either ```qnnpack``` or ```fbgemm```. Please check ```quantize_model.py``` for detailed information.

Test model
-----------------

Command for testing original model:
```sh
python3 test.py --adadelta --valRoot /path/to/test/sample/crnn/dataset  --cuda --pretrained /path/to/exported/model
```
Command for testing exported pruned model:
```sh
python3 test.py --adadelta --valRoot /path/to/test/sample/crnn/dataset  --cuda --finetune --pretrained /path/to/exported/model
```
```test.py```provides method to test different models. 
