from synthetic_image_analyzer.extract_features import compute_feature
from synthetic_image_analyzer.quality_evaluator import ComputeAllMetrics
import numpy as np
from sklearn.preprocessing import normalize


## This file provides examples of computing MM-FID and MM-STD in our TMI paper
ref_dir = './real'
fake_dir = './fake'
class_names = ['PNEUMONIA','NORMAL']
num_channels = 1
save_path = './'


# ckpt_paths = {'ImageNet':None,
#               'Age':'./inception_xray_age.pth',
#             'Sex':'./inception_xray_sex.pth',
#             'Discriminator':'./inception_xray_dis.pth',
#               }

ckpt_paths = {'ImageNet':None,
              }

resolutions = [2**7,2**8,2**9,2**10]

fid_dict = np.zeros((len(ckpt_paths),len(resolutions)))
std_dict = np.zeros((len(ckpt_paths),len(resolutions)))


mm_fid = 0
mm_std = 0

for idx_model,model_name in enumerate(ckpt_paths.keys()):
    for idx_res,resolution in enumerate(resolutions):
        # extract features using different models
        fake_features = compute_feature(fake_dir, class_names, resolution, num_channels, save_path,
                        'inception_v3', ckpt_path=ckpt_paths[model_name], batch_size=16, gpu_ids=0, save_features=True,
                                        save_name=f'fake_{model_name}_{resolution}')
        real_features = compute_feature(ref_dir, class_names, resolution, num_channels, save_path,
                        'inception_v3', ckpt_path=ckpt_paths[model_name], batch_size=16, gpu_ids=0, save_features=True,
                                        save_name=f'real_{model_name}_{resolution}')

        # initialize metrics parameters
        cam = ComputeAllMetrics(real_features, fake_features, fake_dir, class_names, save_path,
                 feature_type='numeric')

        fid_dict[idx_model,idx_res] = cam.fid()
        std_dict[idx_model,idx_res] = cam.std()

fid_dict = normalize(fid_dict)
std_dict = normalize(std_dict)
print('MM-FID on all models is %0.2f (%0.2f)'%(np.mean(fid_dict),np.std(fid_dict)))
print('MM-STD on all models is %0.2f (%0.2f)'%(np.mean(std_dict),np.std(std_dict)))
