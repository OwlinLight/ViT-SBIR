import torch

import dataset
from model import FGSBIR_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50/ ViT')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--nThreads', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)

    hp = parser.parse_args()
    dataset_Train  = dataset.FGSBIR_RGB_Dataset(hp, mode ='Train')
    dataset_Test  = dataset.FGSBIR_RGB_Dataset(hp, mode ='Test')
    # dataloader_Train, dataloader_Test = dataset.get_dataloader_RGB(hp)
    dataset_pic = dataset.FGSBIR_RGB_Pic_Dataset(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)
    model.load_state_dict(torch.load('VGG_ShoeV2_model_best_run.pth', map_location=device))

    model.eval()
    item = 2
    # # pure evaluate

    with torch.no_grad():
        top_1, top_10 = model.get_top(dataset_pic, dataset_Test, item)
        actual = dataset_Test[item]['sketch_path']
    print('actual' + actual)
    print(top_1)
    print(top_10)

    # # pure evaluate
    # with torch.no_grad():
    #     top1_eval, top10_eval = model.evaluate(dataloader_Test)
    #     print('results : ', top1_eval, ' / ', top10_eval)

