from src_files.models.resnet101 import ResNet_CSRA

import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import config
from dateset import TrainDataset, ValDataset
from src_files.helper_functions.helper_functions import ModelEma, \
    add_weight_decay, calc_average_precision, get_roc_auc_score, evaluation
from src_files.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='PyTorch DPNet Training')
parser.add_argument('--dataset', type=str, default='/home/user4/workplace/CSRA/data/images')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

def main():

    args = parser.parse_args()

    Train_dataset = TrainDataset(args.dataset, transform=config.transform_train)

    Val_dataset = ValDataset(args.dataset, transform=config.transform_val)

    print('\n-----Initial Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(Train_dataset)))
    print('num images in val_dataset     : {}'.format(len(Val_dataset)))
    print('-------------------------------------')


    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        Train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    print('\n-----Initial Batchloaders Information -----')
    print('num batches in train_loader: {}'.format(len(train_loader)))
    print('num batches in val_loader  : {}'.format(len(val_loader)))
    print("num Symptom class:", len(Train_dataset.all_classes))
    print("num Pathology class:", len(Train_dataset.all_Pathology_classes))
    print("num val Symptom class:", len(Val_dataset.all_classes))
    print("num val Pathology class:", len(Val_dataset.all_Pathology_classes))
    print('-------------------------------------------')

    model = ResNet_CSRA(num_heads=4, lam=0.4,
                        num_classes=len(Train_dataset.all_Pathology_classes),
                        num_Symptom_classes=len(Train_dataset.all_classes),
                        cutmix=None)

    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1])

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr)


def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 100
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    loss_func = F.mse_loss
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.3)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        for i, (inputData, target,Symptom_target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            Symptom_target = Symptom_target.cuda()
            with autocast():  # mixed precision
                output1,output2,output3,x1,x2 = model(inputData,True)  # sigmoid will be done in loss !
                output1, output2, output3 = output1.float(),output2.float(),output3.float()
                x1,x2 = x1.float(),x2.float()
            loss1 = criterion(output1, target)
            loss2 = criterion(output2, Symptom_target)
            all_target = torch.hstack((target, Symptom_target))
            loss3 = criterion(output3, all_target)
            loss4 = loss_func(x1,x2,reduction="sum")
            loss = 0.01*loss1 + loss2 + 0.01*loss3 + 0.00001*loss4
            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model_best.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input, target,Symptom_target,name) in enumerate(val_loader):
        with torch.no_grad():
            with autocast():
                output_regular,output_regular1 = model(input.cuda(),False)
                output_regular = Sig(output_regular).cpu()


        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(Symptom_target.cpu().detach())

    mAP_score_regular = calc_average_precision(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())

    specificity = dict()
    sensitivity = dict()

    for i in range(38):
        specificity[i],sensitivity[i] = evaluation(torch.cat(targets).numpy()[:, i], torch.cat(preds_regular).numpy()[:, i],3)
    print("symptom_specificity:",specificity)
    print("symptom_sensitivity:", sensitivity)
    mean_specificity = sum(specificity.values())
    mean_specificity = mean_specificity / len(specificity)
    mean_sensitivity = sum(sensitivity.values())
    mean_sensitivity = mean_sensitivity / len(sensitivity)
    print("symptom_mean_specificity:",mean_specificity)
    print("symptom_mean_sensitivity:", mean_sensitivity)
    auc = get_roc_auc_score(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    print("symptom_auc:",auc)
    return mAP_score_regular


if __name__ == '__main__':
    main()
