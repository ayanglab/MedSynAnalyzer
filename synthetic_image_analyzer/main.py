import argparse
from .extract_features import compute_feature
from .quality_evaluator import ComputeAllMetrics


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # file paths
        self.parser.add_argument('-d', "--data_dir", type=str, default='/home/xiaodan/PycharmProjects/privacy_analyzer/demo/fake',
                                 help="File path for the synthetic dataset.")
        self.parser.add_argument('-r', "--ref_dir", type=str, default='/home/xiaodan/PycharmProjects/privacy_analyzer/demo/real',
                                 help="File path for the real dataset for fidelity computation.")
        self.parser.add_argument('-o', "--output_dir", type=str, default='/home/xiaodan/PycharmProjects/privacy_analyzer/demo/',
                                 help="File path for output features and average images.")
        self.parser.add_argument("--resolution", type=int, default=512,
                                 help="The image resolution")
        self.parser.add_argument("--num_channels", type=int, default=1,
                                 help="The number of channels")

        # feature extractor parameter
        self.parser.add_argument("--select_model", type=str, default='vqvae_bottom',
                                 help="Models used for the synthesis. Available models include vqvae_top,"
                                      "vqvqe_bottom and inception_v3.")
        self.parser.add_argument("--pretrained_model_path", type=str, default='/home/xiaodan/PycharmProjects/privacy_analyzer/demo/vqvae_011.pt',
                                 help="Pre-trained models for feature extraction.")
        self.parser.add_argument("--class_names", type=str, default=None,
                                 help="For multiple classes, we provide computation on selected classes.")
        self.parser.add_argument("--gpu_ids", type=int, default=0,
                                 help="GPU ids for feature extraction. Setting this value to -1 to disable the GPU.")

        # quality evaluator parameter
        self.parser.add_argument("--nearest_k", type=int, default=5,
                                 help="The threshold for nearest neighbors.")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt


class SyntheticImageAnalyzer:
    def __init__(self):
        self.opt = BaseOptions().parse()

    def feature_extractor(self):
        opt = self.opt

        fake_features = compute_feature(opt.data_dir, opt.class_names, opt.resolution, opt.num_channels,
                                        opt.output_dir, opt.select_model, opt.pretrained_model_path,
                                        batch_size=16, gpu_ids=opt.gpu_ids, save_features=True)

        real_features = compute_feature(opt.ref_dir, opt.class_names, opt.resolution, opt.num_channels,
                                        opt.output_dir, opt.select_model, opt.pretrained_model_path,
                                        batch_size=16, gpu_ids=opt.gpu_ids, save_features=True)

        return real_features, fake_features

    def compute_qualities(self, real_features, fake_features):
        opt = self.opt

        if 'vqvae' in opt.select_model:
            feature_type = 'discrete'
        else:
            feature_type = 'numeric'

        cam = ComputeAllMetrics(real_features, fake_features, opt.data_dir, opt.class_names, opt.output_dir,
                                feature_type, opt.nearest_k)

        return cam.get_all()

    def inference(self):
        real_features, fake_features = self.feature_extractor()
        metrics = self.compute_qualities(real_features, fake_features)
        for k, v in sorted(metrics.items()):
            print('%s: %s' % (str(k), str(v)))

def main():
    analyzer = SyntheticImageAnalyzer()
    metrics = analyzer.inference()
