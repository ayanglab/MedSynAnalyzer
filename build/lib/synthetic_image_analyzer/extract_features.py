import numpy as np
import pandas as pd
import timm
from .image_dataset import image_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from .vqvae import VQVAE



def compute_feature(filepath, class_names, resolution, num_channels, save_path,
                    model_name, ckpt_path=None, batch_size=16, gpu_ids=0, save_features=True, save_name=None):
    train_dt = image_dataset(filepath=filepath, class_names=class_names, resolution=resolution, num_channels=num_channels)
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

    print('Begin feature extraction.')
    if gpu_ids < 0:
        gpu_ids = 'cpu'

    if 'vqvae' not in model_name:
        encoder = timm.create_model(model_name, pretrained=True)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        encoder = encoder.to(device=gpu_ids)
        encoder.eval()
    else:
        encoder = VQVAE(in_channel=num_channels)
        if ckpt_path is None:
            raise ValueError('The VQVAE path is not specified.')
        encoder.load_state_dict(torch.load(ckpt_path))
        encoder.eval()
        encoder = encoder.to(device=gpu_ids)
        encoder = encoder.encode



    features = []
    filenames = []

    batch = tqdm(train_loader, total=len(train_dt) // batch_size)

    for data in batch:
        images = data['image'].cuda()
        filename = data['filename']
        if num_channels != 3 and 'vqvae' not in model_name:
            images = images.repeat(1, 3, 1, 1)
        if model_name == 'vqvae_top':
            _, _, _, e, _ = encoder(images)
        elif model_name == 'vqvae_bottom':
            _, _, _, _, e = encoder(images)
        else:
            e = encoder(images)
        e = torch.flatten(e,1)
        e = np.array(e.detach().cpu())

        for i in range(images.shape[0]):
            features.append(e[i])
        filenames.append(filename)

    filenames = np.concatenate(filenames, axis=0)
    features_arr = np.asarray(features)
    filenames = np.asarray(filenames)
    features = pd.DataFrame(features_arr)
    features['filename'] = filenames

    print('Feature extraction completed.')

    if not os.path.exists(os.path.join(save_path, 'Features')):
        os.mkdir(os.path.join(save_path, 'Features'))

    if save_features:
        if save_name is None:
            save_name = '%s_%s' % (os.path.split(filepath)[-1], model_name)
        save_name = os.path.join(save_path, 'Features', save_name + '.csv')
        features.to_csv(save_name)
        print(f'Extracted features are saved in {save_name}')

    return np.array(features.drop('filename', axis=1))