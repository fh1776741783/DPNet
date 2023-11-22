from sklearn.metrics import roc_curve, auc

from src_files.models.resnet101 import ResNet_CSRA

import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch import nn


import config
from dateset import TrainDataset, ValDataset
from src_files.helper_functions.helper_functions import calc_average_precision, get_roc_auc_score, evaluation
from torch.cuda.amp import autocast


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch DPNet Training')
parser.add_argument('--dataset', type=str, default='/home/user4/workplace/CSRA/data/image')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-path', default='./model_best.ckpt', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')

def main():

    args = parser.parse_args()

    Val_dataset = ValDataset(args.dataset,transform=config.transform_val)

    print('\n-----Initial Dataset Information-----')
    print('num images in val_dataset     : {}'.format(len(Val_dataset)))
    print('-------------------------------------')


    val_loader = torch.utils.data.DataLoader(
        Val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    print('\n-----Initial Batchloaders Information -----')
    print('num batches in val_loader  : {}'.format(len(val_loader)))
    print("num val Symptom class:", len(Val_dataset.all_classes))
    print("num val Pathology class:", len(Val_dataset.all_Pathology_classes))
    print('-------------------------------------------')

    model = ResNet_CSRA(num_heads=4, lam=0.4,
                        num_classes=len(Val_dataset.all_Pathology_classes),
                        num_Symptom_classes=len(Val_dataset.all_classes),
                        cutmix=None)

    model = nn.DataParallel(model)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.cpu()
    model = model.cuda().half().eval()
    print('done')

    mAP_score = validate_multi(val_loader, model)

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
    fpr = dict()
    tpr = dict()
    FPR = dict()
    roc_auc = dict()
    specificity = dict()
    sensitivity = dict()
    for i in range(38):
        specificity[i],sensitivity[i] = evaluation(torch.cat(targets).numpy()[:, i], torch.cat(preds_regular).numpy()[:, i],3)
        FPR[i] = 1 - specificity[i]
        fpr[i], tpr[i], _ = roc_curve(torch.cat(targets).numpy()[:, i], torch.cat(preds_regular).numpy()[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print("TNR:", specificity)
    print("TPR:", sensitivity)
    print("FPR:", FPR)
    mean_specificity = sum(specificity.values())
    mean_specificity = mean_specificity / len(specificity)
    mean_sensitivity = sum(sensitivity.values())
    mean_sensitivity = mean_sensitivity / len(sensitivity)
    print("symptom_mean_specificity:",mean_specificity)
    print("symptom_mean_sensitivity:", mean_sensitivity)

    symptom_auc = get_roc_auc_score(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    print("symptom_auc:",symptom_auc)
    return mAP_score_regular

if __name__ == '__main__':
    main()
