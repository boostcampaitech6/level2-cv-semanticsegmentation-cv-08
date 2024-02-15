import os
import os.path as osp
import time
import math
from datetime import timedelta, datetime
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

import sys, os

from models.models import get_model
from models.losses import get_loss_function
from models.optimizer import get_optimizer

from modules.utils import load_yaml, save_yaml
from modules.dataset import XRayDataset, XRayInferenceDataset
from modules.schedulers import get_scheduler
from modules.transforms import get_transform_function

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)


def validation(epoch, model, data_loader, criterion, device, thr=0.5, num_classes=29):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = utils.get_classes()
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.to(device), masks.to(device)        
            model = model.to(device)
            
            outputs = model(images)
            
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


def do_training(config):

    # Set train serial: ex) 20211004
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_serial = 'debug' if config['debug'] else train_serial

    # Create train result directory and set logger
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)
    

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set augmentation
    transform = get_transform_function(config['transform'],config)

    train_dataset = XRayDataset(
        is_train=True,
        transforms=transform,
        data_path="data/train",
    )
    valid_dataset = XRayDataset(
        is_train=True,
        transforms=transform,
        data_path="data/train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=False
    )


    model = get_model(model_str=config['architecture'])
    model = model(classes=config['n_classes'],
                encoder_name=config['encoder'],
                encoder_weights=config['encoder_weight'],
                activation=config['activation']
                ,encoder_output_stride=config['stride']).to(device)
    model.to(device)
    
    optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
    optimizer = optimizer(model.parameters(), **config['optimizer']['args'])
    
    scheduler = get_scheduler(scheduler_str=config['scheduler']['name'])
    scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])
    
    criterion = get_loss_function(loss_function_str=config['loss']['name'])
    criterion = criterion(**config['loss']['args'])


    print(f'Start training..')
    
    n_class = config['n_classes']
    best_dice = 0.
    
    for epoch in range(config['max_epoch']):
        model.train()
        epoch_loss = 0
        for step, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)
            
            outputs = model(images)
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss / len(train_loader)
        print(train_loss)
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        
        dice = validation(epoch + 1, model, valid_loader, criterion, device, 0.5, n_class)
        
        if best_dice < dice:
            print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
            print(f"Save model in {train_result_dir}")
            best_dice = dice

            check_point = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None
            }
        
            torch.save(check_point,os.path.join(train_result_dir,f'model_{epoch}.pt'))
            torch.save(check_point,os.path.join(train_result_dir,f'best_model.pt'))
            early_stopping_count = 0
        else:
            early_stopping_count += 1
        if early_stopping_count >= config['earlystopping_patience']:
            exit()
            


if __name__ == '__main__':
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)

    # Set random seed, deterministic
    torch.cuda.manual_seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    do_training(config)