import sys 
sys.path.append("..") 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse

from models.pruned_fots import FOTS_pruned as FOTS_q

import os
import cv2
import torch.optim as optim
import math
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

# Define new conv method
def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=False))
    return nn.Sequential(*modules)

# Add different classification head in forward process.
class clf_FOTS(FOTS_q):
    def __init__(self, crop_height=640):
        super().__init__(crop_height)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.down_conv = conv(64, 64, kernel_size=3, stride=4)
        self.last_conv = nn.Conv2d(64, 64, 1)
        self.fc = nn.Linear(64, 1)
        self.init_fc(self.fc)
        self.sigm = nn.Sigmoid()
        
    def init_fc(self, fc):
        torch.nn.init.normal_(fc.weight, mean=0, std=1)
        torch.nn.init.constant_(fc.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        x = self.last_conv(e2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.down_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigm(x)
        
        return x

class clf_dataloader():
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
        scale_y = 270 / min(image.shape[0],image.shape[1])   # 1248 # 704
        scale_x = scale_y
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        orig_scaled_image = scaled_image.copy()

        scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.squeeze(torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float())

        label = torch.tensor(int(self.img_list[idx].split('/')[-1].split('_')[1]))
        return image_tensor, label
    
def train_clf(model, dataloaders, criterion, optimizer, num_epoch=25):
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch-1))
        print('-'*10)
        
        for phase in ['train', 'test']:
            
            #Calculate ROC curve and AUC score
            y_true = np.empty(0)
            y_pred = np.empty(0)
            
            if phase == 'trian':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.unsqueeze(labels, 1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())
                    mask = outputs > 0.5
                    preds = torch.zeros(outputs.size(), dtype=torch.int32)
                    for i in range(list(preds.size())[0]):
                        if outputs[i].item() > 0.5:
                            preds[i] = 1
                    # Code for auc calcaulating
                    output_np = outputs.squeeze().cpu().detach().numpy()
                    label_np = labels.data.squeeze().cpu().detach().numpy()
                    if output_np.ndim == 0:
                        output_np = np.expand_dims(output_np, axis=0)
                    if label_np.ndim == 0:
                        label_np = np.expand_dims(label_np, axis=0)
                    y_pred = np.concatenate((y_pred, output_np))
                    y_true = np.concatenate((y_true, label_np))
                    
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                for i in range(list(preds.size())[0]):
                    if str(preds[i].item()) == str(labels.data[i].item()):
                        running_corrects += 1
            
            # Calculate ROC and AUC
            lr_fpr, lr_tpr, thrs = roc_curve(y_true, y_pred)
            auc = metrics.auc(lr_fpr, lr_tpr) 
            for i in range(y_pred.shape[0]):
                if y_pred[i] > 0.5:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} AUC: {:.4f} Recall: {:.4f} Precision: {:.4f}'.format(phase, epoch_loss,\
                                                                    auc, recall, precision))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rejector-folder', type=str, required=True, help='Path to folder with rejector images and labels')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    args = parser.parse_args()
        

    # Load pretrained model
    clf = clf_FOTS()

    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
#     clf.load_state_dict(checkpoint['model_state_dict'], strict=False)
    clf.load_state_dict(checkpoint, strict=False)

    need_grads = ['fc.weight','fc.bias','last_conv.weight','last_conv.bias',\
             'down_conv.0.weight', 'down_conv.0.bias', 'down_conv.1.weight', 'down_conv.1.bias']

    # Set parameters whether need grads.
    for name, param in clf.named_parameters():
        print(name, ": ", param.requires_grad)
    #     if name == 'fc.weight' or name == 'fc.bias' or name == 'last_conv.weight' or name == 'last_conv.bias':
        if name in need_grads:
            print("After setting: ", name, ": ", param.requires_grad)
            continue
        param.requires_grad = False
        print("After setting: ", name, ": ", param.requires_grad)

    _dummy_input_data = torch.rand(1, 3, 299, 299).cpu()
    res = clf(_dummy_input_data)
    
    # Dataloader for classifition

   
    img_list = []
    # I got these datasets by running FOTS model. If there is one or more detect bounary boxes, set label 1, otherwise label 0.
    paths = [args.rejector_folder]
    for path in paths:
        for file in os.listdir(path):
            if file.startswith('base02'):
                continue
            file_path = os.path.join(path, file)
            img_list.append(file_path)
    print(len(img_list))
    img_list_test = []
    paths_test = [args.rejector_folder]
    for path in paths_test:
        for file in os.listdir(path):
            if file.startswith('base02'):
                file_path = os.path.join(path, file)
                img_list_test.append(file_path)
    print(len(img_list_test))
    img_train_loader = clf_dataloader(img_list)
    img_test_loader = clf_dataloader(img_list_test)
    dataloaders = {'train':torch.utils.data.DataLoader(img_train_loader, batch_size=1, shuffle=True, num_workers=1),
                   'test':torch.utils.data.DataLoader(img_test_loader, batch_size=1, shuffle=True, num_workers=1)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set optimizer and criterion
    params_to_update = []
    for name, params in clf.named_parameters():
        if params.requires_grad == True:
            params_to_update.append(params)
            print('{} need grad.'.format(name))
    optimizer_ft = optim.SGD(params_to_update, lr=0.002, momentum=0.9)
    criterion = nn.BCELoss()
    
    clf = clf.to(device)

    clf_trained = train_clf(clf, dataloaders, criterion, optimizer_ft, num_epoch=args.epochs)
    
    torch.save(clf.state_dict(), '{}/early_exit_model.pkl'.format(args.save_dir))
