# Code for l1 0.5
import copy
import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.utils.data
import tqdm
import torch.quantization._numeric_suite as ns
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.quanted_fots import FOTS_quanted
from data import datasets
from utils.train_utils import detection_loss
from quantize_model import rename_state_dict_keys, compute_error, pre_process, post_process


def val_one_epoch(model, loss_func, loader_val, device):
    model.eval()
    model.to(device)
    test_loss_stats = 0.0
    val_loss = 0.0
    val_loss_count = 0

    pbar = tqdm.tqdm(loader_val, 'Val result: ', ncols=80)
    cnt_error = 0
    loss = 0
    for cropped, classification, regression, thetas, training_mask in pbar:
        cropped = cropped.to(device)
        classification = classification.to(device)
        regression = regression.to(regression)
        thetas = thetas.to(device)
        training_mask = training_mask.to(device)
        prediction = model(cropped)
        if prediction[0].shape[-1] != classification.shape[-1] or \
                prediction[0].shape[-2] != classification.shape[-2]:
            cnt_error += 1
            continue
        try:
            loss = loss_func(prediction, (classification, regression, thetas, training_mask))
        except RuntimeError as e:
            print(e.message)

        val_loss += loss.item()
        test_loss_stats += loss.item()
        val_loss_count += len(cropped)
    val_loss /= val_loss_count

    return val_loss, cnt_error


def train_model(model, train_loader, val_loader, device):
    model.to(device)
    train_loss_stats = 0.0
    loss_count_stats = 0
    batch_per_iter_cnt = 0

    criterion = detection_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32,
                                                              verbose=True, threshold=0.05, threshold_mode='rel')

    for epoch in range(0, 10):
        model.train()
        Pbar = tqdm.tqdm(train_loader, 'Epoch ' + str(epoch), ncols=80)
        train_loss = 0
        for cropped, classification, regression, thetas, training_mask in Pbar:
            cropped = cropped.to(device)
            classification = classification.to(device)
            regression = regression.to(regression)
            thetas = thetas.to(device)
            training_mask = training_mask.to(device)
            if batch_per_iter_cnt == 0:
                optimizer.zero_grad()
            prediction = model(cropped)

            loss = criterion(prediction, (classification, regression, thetas, training_mask)) / 2
            train_loss_stats += loss.item()
            loss.backward()
            batch_per_iter_cnt += 1
            if batch_per_iter_cnt == 2:
                optimizer.step()
                batch_per_iter_cnt = 0
                loss_count_stats += 1
                train_loss = train_loss_stats / loss_count_stats
                Pbar.set_postfix({'Train loss': f'{train_loss:.5f}'}, refresh=False)

        lr_scheduler.step(train_loss, epoch)
        val_loss, cnt_error = val_one_epoch(model, criterion, val_loader, device)
        print("Epoch: {:03d} Train Loss: {:.3f} Validation Loss: {:.3f} Validation Count Error: {:.3f}"
              .format(epoch + 1, train_loss, val_loss, cnt_error))


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 299, 299)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x)[0].detach().cpu().numpy()
        y2 = model_2(x)[0].detach().cpu().numpy()
        if not np.allclose(a=y1, b=y2, rtol=rtol, atol=atol,
                           equal_nan=False):
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder-syn', type=str, required=True,
                        help='Path to folder with syntext train images and labels')
    parser.add_argument('--train-folder-sample', type=str, default=None,
                        help='Path to folder with sample train images and labels')
    parser.add_argument('--calibrate-folder', type=str, required=True,
                        help='Path to folder with images for calibrating')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='Path to saved model dir')
    parser.add_argument('--pretrain-model', type=str, default=None, help='Pretrained model name in save dir')
    parser.add_argument('--backends', type=str, default='fbgemm', help='Quantization backends')
    args = parser.parse_args()
    images_folder = args.calibrate_folder

    net = FOTS_quanted()

    # Device
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")

    # Load model state dict
    checkpoint_name = args.pretrain_model
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    new_state_dict = rename_state_dict_keys(checkpoint)

    net.load_state_dict(new_state_dict, strict=True)
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    fp_net = copy.deepcopy(net)

    # Must set train for QAT before model fusion
    net.train()
    fp_net.train()

    # Set quantization configure to all parts in FOTs model
    qcf = torch.quantization.get_default_qat_qconfig(args.backends)
    net.part1.qconfig = qcf
    net.part2.qconfig = qcf
    # Fuse all parts in the FOTs model
    net.fuse_model()
    # Check model equivalent for model and fused model
    net.eval()
    fp_net.eval()
    assert model_equivalence(
        model_1=net,
        model_2=fp_net,
        device=cpu_device,
        rtol=1e-03,
        atol=1e-06,
        num_tests=100,
        input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"

    net.train()

    # Prepare the quantization for all parts
    torch.quantization.prepare_qat(net.part1, inplace=True)
    torch.quantization.prepare_qat(net.part2, inplace=True)

    # Prepare data-loader for training and validation
    dataset_train = datasets.MergeText(args.train_folder_syn, args.train_folder_sample, datasets.transform, train=True)
    dataset_val = datasets.MergeText(args.train_folder_syn, args.train_folder_sample, datasets.transform, train=False)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4)

    # Train model
    print("Training QAT Model...")
    train_model(net, train_loader, val_loader, gpu_device)
    net.to(cpu_device)

    # Convert all parts into quantized ones
    net.eval()
    quantized_part1 = torch.quantization.convert(net.part1, inplace=True)
    quantized_part2 = torch.quantization.convert(net.part2, inplace=True)

    # Use some images from the folder for evaluation or calibration
    pbar = tqdm.tqdm(os.listdir(images_folder), desc='Test', ncols=80)
    for image_name in pbar:
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        image_tensor, scale = pre_process(image)
        confidence, distances, angle = net(image_tensor)
        boxes = post_process(confidence, distances, angle, scale)
        for i in range(0, len(boxes)):
            image = cv2.polylines(image, [boxes[i].reshape((-1, 1, 2))], True, (255, 0, 0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('/home/johnhu/Pictures/CVPRW/', image_name), image)

    # Compare each layer of part 1 and 2, the higher score the better
    wt_compare_dict_1 = ns.compare_weights(fp_net.part1.state_dict(), quantized_part1.state_dict())
    wt_compare_dict_2 = ns.compare_weights(fp_net.part2.state_dict(), quantized_part2.state_dict())

    for key in wt_compare_dict_1:
        print(key, compute_error(wt_compare_dict_1[key]['float'], wt_compare_dict_1[key]['quantized'].dequantize()))
    for key in wt_compare_dict_2:
        print(key, compute_error(wt_compare_dict_2[key]['float'], wt_compare_dict_2[key]['quantized'].dequantize()))

    # Get the backend and save the torchscript files to the specific path
    backend = args.backends
    torch.jit.save(torch.jit.script(quantized_part1), '{}/part1_{}.torchscript'.format(args.save_dir, backend))
    torch.jit.save(torch.jit.script(quantized_part2), '{}/part2_{}.torchscript'.format(args.save_dir, backend))
