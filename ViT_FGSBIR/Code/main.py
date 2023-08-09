import torch
import time
from .model import FGSBIR_Model
from .dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='ViT', help='VGG / InceptionV3/ Resnet50/ ViT')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    writer = SummaryWriter(comment = '_' + hp.backbone_name + '_' + hp.dataset_name + '_model')

    model = FGSBIR_Model(hp)
    model.to(device)
    # model.load_state_dict(torch.load('VGG_ShoeV2_model_best.pth', map_location=device))
    # model.load_state_dict(torch.load(hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth', map_location=device))

    step_count, top1, top10 = -1, 0, 0
    loss = 0

    # # pure evaluate
    # with torch.no_grad():
    #     # top1_eval, top10_eval = model.evaluate(dataloader_Test)
    #     model.Image_Feature_Final_All(dataloader_Test)
    #     top1_eval, top10_eval = model.get_top('./../../datasets/ShoeV2/testA/2450845009_1.png')
    #     print('results : ', top1_eval, ' / ', top10_eval)


    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, top1, top10, time.time()-start))
                writer.add_scalar("Loss/train", loss, step_count)


            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)
            
                if top1_eval > top1:
                    torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')
        else:
            writer.add_scalar("Loss/test", loss, i_epoch)
            writer.add_scalar("top1/test", top1, i_epoch)
            writer.add_scalar("top10/test", top10, i_epoch)
    else:
        torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best_run.pth')
        writer.close()