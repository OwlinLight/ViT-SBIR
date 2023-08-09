import sys
sys.path.append('../')

import torch
import ViT_FGSBIR.Code.dataset as dataset
from ViT_FGSBIR.Code.model import FGSBIR_Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

class SBIR:
    def initialize_model(self):
        parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')
        parser.add_argument('--dataset_name', type=str, default='ShoeV2')
        parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50/ ViT')
        parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                            help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
        # from ./../ to ./ (as flask run in tutorial)
        parser.add_argument('--root_dir', type=str, default='./')
        parser.add_argument('--batchsize', type=int, default=1)
        parser.add_argument('--nThreads', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--max_epoch', type=int, default=200)
        parser.add_argument('--eval_freq_iter', type=int, default=100)
        parser.add_argument('--print_freq_iter', type=int, default=1)
        hp, _ = parser.parse_known_args()
        # dataloader_Train, dataloader_Test = dataset.get_dataloader_RGB(hp)
        self.dataset_pic = dataset.FGSBIR_RGB_Pic_Dataset(hp)

        model = FGSBIR_Model(hp)
        model.to(device)
        model.load_state_dict(torch.load(hp.root_dir + 'VGG_ShoeV2_model_best_run.pth', map_location=device))
        # model.load_state_dict(torch.load(hp.root_dir + 'ViT_ShoeV2_model_best.pth', map_location=device))
        self.model = model
        self.hp = hp

    def get_result(self, des = 'uploads', sketch_name = "2429245009_1"):
        self.model.eval()
        # # pure evaluate
        dataset_Test = dataset.FGSBIR_RGB_Dataset(self.hp, mode='Test')

        with torch.no_grad():
            print('getting result...')
            top_10 = self.model.get_top(self.dataset_pic, self.dataset_pic.get_img(des, sketch_name))
            actual = sketch_name.split('_')[0]
        return top_10

