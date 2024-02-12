from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import pdb
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from model_4params import CNN_nopos as CNN
# from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from dataset_posenc import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler

# from tensorboardX import SummaryWriter 
# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/train_diff')
# writerval = SummaryWriter('runs_val/val_diff')

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='train_sub.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()

# def loss_func(norm, init, pred):
#     inter = torch.add(init, pred)
    # pdb.set_trace()

def cos_angle(v1, v2):
    v1 = nn.functional.normalize(v1, dim=1)
    v2 = nn.functional.normalize(v2, dim=1)
    # pdb.set_trace()
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)
    
def loss_func(norm, init, pred):
    pred_norm = torch.add(init, pred)
    loss = (1-cos_angle(pred_norm, norm)).pow(2).mean() 
    # pdb.set_trace()
    return loss

def rms_angular_error(estimated_normals, ground_truth_normals):
    estimated_normals = F.normalize(estimated_normals, dim=1)
    ground_truth_normals = F.normalize(ground_truth_normals, dim=1)

    dot_product = torch.sum(estimated_normals * ground_truth_normals, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angular_diff = torch.acos(dot_product) * torch.div(180.0, torch.pi)
    squared_diff = angular_diff.pow(2)
    mean_squared_diff = torch.mean(squared_diff)
    rms_angular_error = torch.sqrt(mean_squared_diff)

    return rms_angular_error.item()  

def train_pcpnet(opt):

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    target_features = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    import time
    
    # create train and test dataset loaders
    # pdb.set_trace()
    train_dataset = PointcloudPatchDataset(
        task = 'mse_6_4params_no_pos',
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    
    print(len(train_dataset))

    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)
    
    print(len(train_datasampler))        

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))
    
    for i in train_dataloader:
        print(len(i[0]))
        break
    
    test_dataset = PointcloudPatchDataset(
        task = 'mse_6_4params_no_pos',
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    test_datasampler = RandomPointcloudPatchSampler(
        test_dataset,
        patches_per_shape=100,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    print(len(test_datasampler))    
    
    model = CNN()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    #sgd optim
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30], gamma=0.5) # milestones in number of optimizer iterations
    tr_loss_per_epoch = []
    val_loss_per_epoch = []
    mse_loss = nn.MSELoss()

    for epoch in range(opt.nepoch):
        train_loss = []
        val_loss = []
        train_rmse = []
        val_rmse = []
        ori_tr = []
        ori_val = []
        model.train()
        pbar = tqdm(train_dataloader)
        for i, (data, init, params) in enumerate(pbar, 0):
            inputs = data.float()
            inputs = inputs.permute(0,3,1,2)
            params = params.float()
            init = init.float()
            inputs, params, init = inputs.to(device), params.to(device), init.to(device)
            optimizer.zero_grad()
            # inputs = 
            out = model(inputs, init)
            loss = mse_loss(out, params)
            # pdb.set_trace()

            loss.backward()


            prev_param_values = {name: p.clone().detach() for name, p in model.named_parameters()}

            optimizer.step()
            # print("max_ori: ", torch.max(norms))
            # print("min_ori: ", torch.min(norms))
            # print("max_pred: ", torch.max(out))             
            # print("min_pred: ", torch.min(out))
            # print("max_inp: ", torch.max(inputs))
            # print("min_inp: ", torch.min(inputs))
            # print(inputs.shape)
            tolerance = 1e-6  # Define a small tolerance value
            for name, p in model.named_parameters():
                diff = torch.abs(prev_param_values[name] - p.detach())
                max_diff = torch.max(diff)
                if max_diff <= tolerance:
                    print(f"{name} is not updating significantly (max diff: {max_diff.item()})")
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm(2)
                    print(f"{name}: gradient norm = {grad_norm}")
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("Data contains NaN or inf values")
                break

            train_loss.append(loss.item())
            pbar.set_postfix(Epoch=epoch, tr_loss=loss.item())
            pbar.set_description('IterL: {}'.format(loss.item()))

        # bef_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()
        # aft_lr = optimizer.param_groups[0]['lr']
        # if(bef_lr != aft_lr):
        #     print(f'epoch: {epoch}, learning rate: {bef_lr} -> {aft_lr}')

        tot_train_loss = np.mean(train_loss)  
        tr_loss_per_epoch.append(tot_train_loss)

    
        with torch.no_grad():
            model.eval()
            pbar1 = tqdm(test_dataloader)
            for i, (data, init, params) in enumerate(pbar1, 0):
                inputs = data.float()
                inputs = inputs.permute(0,3,1,2)
                params = params.float()
                init = init.float()
                inputs, params, init = inputs.to(device), params.to(device), init.to(device)
                
                out = model(inputs, init)
                loss = mse_loss(out, params)
                val_loss.append(loss.item())
                pbar1.set_postfix(Epoch=epoch, val_loss=loss.item())
                
        tot_val_loss = np.mean(val_loss)
        val_loss_per_epoch.append(tot_val_loss)

        if epoch % 10 == 0:
            EPOCH = epoch
            PATH = "mse_6_4params_no_pos.pt"
            LOSS = tot_train_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        'batchsize' : opt.batchSize,
                        'val_losses_so_far' : val_loss_per_epoch,
                        'train_losses_so_far' : tr_loss_per_epoch
                        }, PATH)
            print("Model saved at epoch: ", epoch)

        print(f'epoch: {epoch} training loss: {tot_train_loss}, {PATH}')
        print(f'epoch: {epoch} val loss: {tot_val_loss}')

        # writer.add_scalar('train loss',tot_train_loss, epoch)
        # writerval.add_scalar('val loss',
        #                     tot_val_loss,
        #                     epoch)


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)


