import numpy as np
import pandas as pd
import os
import cv2
from .utils import compute_fidelity_privacy
from .fid_score import calculate_frechet_distance


def get_data(save_name_real, save_name_fake):
    real_features = pd.read_csv(save_name_real, index_col=0)
    real_features = np.array(real_features.drop('filename', axis=1))
    fake_features = pd.read_csv(save_name_fake, index_col=0)
    fake_features = np.array(fake_features.drop('filename', axis=1))
    return real_features, fake_features


class ComputeAllMetrics:
    def __init__(self, real_features, fake_features, fake_image_path, class_names=None, save_path='./',
                 feature_type='numeric', nearest_k=5, mode_collapse_threshold=0.05):
        """
        Initialize the ComputeAllMetrics class.

        :param real_features: Array of real features.
        :param fake_features: Array of fake features.
        :param fake_image_path: Path to the directory containing fake images.
        :param class_names: List of class names. Default is None.
        :param save_path: Path to save the average images. Default is './'.
        :param feature_type: The input feature type. Default is 'numeric'. For VQVAE features, the input feature type
                             should be 'discrete'.
        :param nearest_k: The threshold value for fidelity computation. Default is 5.
        :param mode_collapse_threshold: The threshold value for mode collapse. Default is 0.05.
        """
        self.real_features = real_features
        self.fake_features = fake_features
        self.fake_image_path = fake_image_path
        self.class_names = class_names
        self.save_path = save_path
        self.feature_type = feature_type
        self.nearest_k = nearest_k
        self.mode_collapse_threshold = mode_collapse_threshold
        self.metrics = {}

    def fidelity_and_privacy(self):
        """
        Compute fidelity and privacy metrics.
        """
        self.f_p_metrics = compute_fidelity_privacy(
            real_features=self.real_features,
            fake_features=self.fake_features,
            nearest_k=self.nearest_k,
            feature_type=self.feature_type,
            mode_collapse_threshold = self.mode_collapse_threshold
        )
        self.metrics.update(self.f_p_metrics)

    def fid(self):
        """
        Compute the Frechet Inception Distance (FID) metric.
        """
        mu = np.mean(self.real_features, axis=0)
        sigma = np.cov(self.real_features, rowvar=False)
        mu_fake = np.mean(self.fake_features, axis=0)
        sigma_fake = np.cov(self.fake_features, rowvar=False)
        if self.feature_type == 'numeric':
            fid = calculate_frechet_distance(mu_fake, sigma_fake, mu, sigma)
            self.metrics['fid'] = fid
        else:
            self.metrics['fid'] = 'FID on discrete values has no meanings'

        return fid

    def std(self):
        """
        Compute the standard deviation of the fake features.
        """
        std_fake = np.mean(np.std(self.fake_features,axis=0))
        self.metrics['std'] = std_fake

        return std_fake

    def filesize(self):
        """
        Compute the average image filesize.
        """
        avg_image_path = self.compute_avg_images()
        file_stats = os.stat(avg_image_path)
        file_size = int(file_stats.st_size / 1024)
        self.metrics['filesize'] = f'{file_size} kB'

        return file_size

    def get_all(self):
        print('Begin computing all metrics.')

        self.fidelity_and_privacy()
        print('Fidelity and privacy metrics completed.')

        _ = self.fid()
        print('FID completed.')

        _ = self.std()
        print('STD completed.')

        _ = self.filesize()
        print('File size for average images completed.')
        return self.metrics

    def compute_avg_images(self):
        """
        Compute the average images per class and save them.
        """
        if self.class_names is None:
            class_names = os.listdir(self.fake_image_path)
        else:
            class_names = self.class_names

        if not os.path.exists(os.path.join(self.save_path, 'Avgs')):
            os.mkdir(os.path.join(self.save_path, 'Avgs'))

        avg_image_list = []
        for label in class_names:
            class_path = os.path.join(self.fake_image_path, label)
            image_files = os.listdir(class_path)

            avg_images_per_class = None
            for idx, file in enumerate(image_files):
                img = cv2.imread(os.path.join(class_path, file))

                if avg_images_per_class is None:
                    c, h, w = img.shape
                    avg_images_per_class = np.zeros([c, h, w], dtype=np.float32)

                avg_images_per_class += img

            avg_images_per_class /= (idx + 1)
            avg_images_per_class = avg_images_per_class.astype(np.uint8)
            avg_image_list.append(avg_images_per_class)

            # Save average image for each class as PNG (fully lossless)
            filename = os.path.split(self.fake_image_path)[-1]
            avg_image_filename = f'{filename}_{label}.png'
            avg_image_path = os.path.join(self.save_path, 'Avgs', avg_image_filename)
            cv2.imwrite(avg_image_path, avg_images_per_class)

        avg_image = np.mean(avg_image_list, axis=0).astype(np.uint8)
        all_classes_avg_image_filename = f'{filename}_all_classes.png'
        all_classes_avg_image_path = os.path.join(self.save_path, 'Avgs', all_classes_avg_image_filename)
        cv2.imwrite(all_classes_avg_image_path, avg_image)

        return all_classes_avg_image_path
