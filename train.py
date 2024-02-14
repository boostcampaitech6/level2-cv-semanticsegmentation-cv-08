import os
import os.path as osp
import time
import math
import datetime
from datetime import timedelta
from argparse import ArgumentParser

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

import utils
from dataset import XRayDataset, XRayInferenceDataset
from model import fcn_resnet50



def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default='data/train')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', default=47)
    parser.add_argument('--num_classes', type=int, default=29)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def validation(epoch, model, data_loader, criterion, thr=0.5, num_classes=29):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = utils.get_classes()
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = utils.dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(n_class, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, train_loader, val_loader, criterion, optimizer, num_classes, max_epoch, model_dir):
    print(f'Start training..')
    
    n_class = num_classes
    best_dice = 0.
    
    for epoch in range(max_epoch):
        model.train()

        for step, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)['out']
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{max_epoch}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % 20 == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, 0.5, n_class)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {model_dir}")
                best_dice = dice
                utils.save_model(model, file_name='fcn_resnet50_best_model.pt', model_dir=model_dir)

def do_training(data_dir, model_dir, device, num_workers, output_dir, image_size,
                input_size, batch_size, learning_rate, max_epoch, save_interval, seed, num_classes):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set augmentation
    transform = A.Compose([
        A.Resize(input_size, input_size),
    ])

    train_dataset = XRayDataset(
        is_train=True,
        transforms=transform,
        data_path=data_dir,
    )
    valid_dataset = XRayDataset(
        is_train=True,
        transforms=transform,
        data_path=data_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    # seed 고정
    utils.set_seed(seed)

    model = fcn_resnet50()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

    train(model, train_loader, valid_loader, criterion, optimizer, num_classes, max_epoch, model_dir)

def main(args):


    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)