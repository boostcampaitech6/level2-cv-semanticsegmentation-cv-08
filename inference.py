
import os
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from modules import utils
import model

from models.models import get_model

from modules.utils import load_yaml, save_yaml
from modules.dataset import XRayInferenceDataset
from modules.transforms import get_transform_function

import random, os, sys
import numpy as np
from datetime import datetime


prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

def test(model, data_loader,device, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = 29
        IND2CLASS = utils.ind2class()
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.to(device) 
            outputs = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = utils.encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


if __name__ == '__main__':

    config = load_yaml(os.path.join(prj_dir, 'config', 'test.yaml'))
    train_config = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'train.yaml'))
   
    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #Device set
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #result_dir
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_serial)
    os.makedirs(pred_result_dir, exist_ok=True)

    transform = get_transform_function(train_config['transform'],train_config)

    model = get_model(model_str=train_config['architecture'])
    model = model(
        **train_config['model_args']
    ).to(device)
    
    # model = model(classes=train_config['n_classes'],
    #             encoder_name=train_config['encoder'],
    #             encoder_weights=train_config['encoder_weight'],
    #             # activation=train_config['activation'],
    #             # encoder_output_stride=train_config['stride']
    #             ).to(device)

    check_point_path = os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'best_model.pt')
    check_point = torch.load(check_point_path,map_location=torch.device("cpu"))
    model.load_state_dict(check_point['model'])

    # Save config
    save_yaml(os.path.join(pred_result_dir, 'train.yaml'), train_config)
    save_yaml(os.path.join(pred_result_dir, 'predict.yaml'), config)
    
    
    data_dir = config['test_dir']

    test_dataset = XRayInferenceDataset(transforms=transform, data_path=data_dir)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )


    rles, filename_and_class = test(model, test_loader,device)

    utils.save_csv(rles, filename_and_class, os.path.join(pred_result_dir))
