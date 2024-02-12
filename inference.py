
import os
import torch
import tqdm
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import Dataset, DataLoader

import utils
import model
from dataset import XRayInferenceDataset

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = 29
        IND2CLASS = utils.ind2class()
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = utils.encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class



model_path = "trained_models"
model_name = "fcn_resnet50"
model = torch.load(os.path.join(model_path, model_name, "_best_model.pt"))

tf = A.resize(512, 512)
test_dataset = XRayInferenceDataset(transforms=tf)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)
rles, filename_and_class = test(model, test_loader)

utils.save_csv(rles, filename_and_class, f'outputs/{model_name}')