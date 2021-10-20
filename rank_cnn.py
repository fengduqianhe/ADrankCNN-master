# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/3/7 
# version： Python 3.7.8
# @File : rank_cnn.py
# @Software: PyCharm
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from adni import AdniDataSet
from sklearn.model_selection import train_test_split
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [6]))
start = datetime.now()
seed = 0
torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 31)

        self.fc_branch_list = nn.ModuleList([nn.Linear(512, 2) for i in range(30)])

    def forward(self, x1):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.shape[0], -1)
        merged = x1
        line = self.fc4(x1)
        line = self.fc5(line)

        branch = []
        for i_num in range(0, 30):
            single_branch = self.fc_branch_list[i_num].forward(x1.unsqueeze(1))
            branch.append(single_branch)
        branch = torch.cat(branch, dim=1)

        # merged = F.normalize(merged, 1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        x1 = x1.view(x1.shape[0])
        return x1, merged, branch, line


if __name__ == "__main__":
    NUM_CHNS = 1
    FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
    NUM_REGION_FEATURES = 64
    NUM_SUBJECT_FEATURES = 64

    WITH_BN = True
    WITH_DROPOUT = True
    DROP_PROB = 0.5

    SHARING_WEIGHTS = False

    PATCH_SIZE = 25

    TRN_BATCH_SIZE = 5
    TST_BATCH_SIZE = 5
    IMAGE_SIZE = 90
    NUM_EPOCHS = 100

    # IMAGE1_PATH = "/BL818_processed/"
    # IMAGE2_PATH = "/BL776_processed/"
    #
    # ADNI1_DATA = pd.read_csv("/ADNIMERGE_ADNI1_BL_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/ADNIMERGE_ADNI2_BL_PROCESSED.csv")

    IMAGE2_PATH = "/BL818_processed/"
    IMAGE1_PATH = "/BL776_processed/"

    ADNI2_DATA = pd.read_csv("/ADNIMERGE_ADNI1_BL_PROCESSED.csv")
    ADNI1_DATA = pd.read_csv("/ADNIMERGE_ADNI2_BL_PROCESSED.csv")

    TRN_LBLS = ADNI1_DATA['MMSE'].tolist()
    VAL_LBLS = ADNI2_DATA['MMSE'].tolist()
    TRN_SUBJECT_IDXS = ADNI1_DATA['SID'].tolist()
    VAL_SUBJECT_IDXS = ADNI2_DATA['SID'].tolist()
    TRN_TPS = ADNI1_DATA['DX_bl'].tolist()
    VAL_TPS = ADNI2_DATA['DX_bl'].tolist()

    print(len(TRN_SUBJECT_IDXS))
    print(len(VAL_SUBJECT_IDXS))
    TRN_STEPS = int(np.round(len(TRN_SUBJECT_IDXS) / TRN_BATCH_SIZE))
    TST_STEPS = int(np.round(len(VAL_SUBJECT_IDXS) / TST_BATCH_SIZE))

    train_subject_num = len(TRN_SUBJECT_IDXS)
    val_subject_num = len(VAL_SUBJECT_IDXS)

    train_flow = AdniDataSet(IMAGE1_PATH, TRN_SUBJECT_IDXS, TRN_LBLS, TRN_TPS, IMAGE_SIZE)
    test_flow = AdniDataSet(IMAGE2_PATH, VAL_SUBJECT_IDXS, VAL_LBLS, VAL_TPS, IMAGE_SIZE)

    train_loader = DataLoader(dataset=train_flow, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=test_flow, batch_size=12, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=70, verbose=True)
    model.to(device)
    result_list = []
    epochs = 150

    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct = 0
        total = 0
        running_loss = 0
        running_loss_tripet = 0
        running_loss_pair = 0
        running_loss_branch = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels, group, labels_line, branch = data
            inputs, labels, group, labels_line, branch = inputs.to(device), labels.to(device), group.to(
                device), labels_line.to(
                device), branch.to(device)
            optimizer.zero_grad()
            logps, merged, out_branch, outline = model.forward(inputs)

            '''branch_loss'''
            loss_branch = torch.tensor(0).float().to(device)
            for i in range(30):
                loss_branch += F.cross_entropy(out_branch[:, i], branch[:, i].long())

            print("loss_branch", loss_branch)

            bt_size = merged.shape[0]
            if bt_size % 2 == 0:
                merged1 = merged[:int(bt_size / 2), :]
                merged2 = merged[int(bt_size / 2):, :]
                group1 = group[:int(bt_size / 2)]
                group2 = group[int(bt_size / 2):]
                labels1 = labels[:int(bt_size / 2)]
                labels2 = labels[int(bt_size / 2):]
                logps1 = logps[:int(bt_size / 2)]
                logps2 = logps[int(bt_size / 2):]
            else:
                merged1 = merged[:int(bt_size / 2) + 1, :]
                merged2 = merged[int(bt_size / 2):, :]
                group1 = group[:int(bt_size / 2) + 1]
                group2 = group[int(bt_size / 2):]
                labels1 = labels[:int(bt_size / 2) + 1]
                labels2 = labels[int(bt_size / 2):]
                logps1 = logps[:int(bt_size / 2) + 1]
                logps2 = logps[int(bt_size / 2):]

            # group_compare = (torch.abs(labels1 - labels2) <= 3)
            logps_sig = torch.sigmoid(logps1 - logps2)

            group_compare = torch.tensor(np.zeros(labels1.shape[0])).float().to(device)
            group_compare[torch.abs(labels1 - labels2) <= 5] = 0.5
            group_compare[labels1 - labels2 < -5] = 0
            group_compare[labels1 - labels2 > 5] = 1

        
            loss_pair = F.binary_cross_entropy(logps_sig, group_compare)


            loss_mmse = F.mse_loss(logps, labels.float())
            loss_line = bce(outline, labels_line.float())

            print("loss_mmse", loss_mmse)
            # print("loss_line", loss_line)
            # print("loss_pair", loss_pair)
            # print("loss_branch", loss_branch)

            loss_mmse = 0 * loss_line + 1 * loss_mmse + 0 * loss_pair + 0 * loss_branch
            loss_mmse.backward()
            optimizer.step()
            running_loss += loss_mmse.item()
            running_loss_tripet += loss_line
            running_loss_pair += loss_pair
            running_loss_branch += loss_branch

        train_loss = running_loss / len(train_loader)
        tripet_loss = running_loss_tripet / len(train_loader)
        pair_loss = running_loss_pair / len(train_loader)
        branch_loss = running_loss_branch / len(train_loader)
        print('Epoch[{}/{}], train_loss:{:.4f}'.format(epoch + 1, epochs, train_loss))
        # print('Epoch[{}/{}], tripet_loss:{:.4f}'.format(epoch + 1, epochs, tripet_loss))
        # print('Epoch[{}/{}], pair_loss:{:.4f}'.format(epoch + 1, epochs, pair_loss))
        # print('Epoch[{}/{}], branch_loss:{:.4f}'.format(epoch + 1, epochs, branch_loss))

        val_running_loss = 0
        val_branch_loss = 0
        val_line_loss = 0

        correct = 0
        total = 0
        classnum = 2
        model.eval()

        cc_label = []
        cc_predict = []
        with torch.no_grad():
            print("validation...")
            for data in val_loader:
                inputs, labels, group, labels_line, branch = data
                inputs, labels, group, labels_line, branch = inputs.to(device), labels.to(device), group.to(
                    device), labels_line.to(
                    device), branch.to(device)

                logps, merged, out_branch, outline = model.forward(inputs)
                print("logps", logps)
                print("labels", labels.float())
                loss_mmse = F.mse_loss(logps, labels.float())
                val_running_loss += loss_mmse.item()

                pre_labels = torch.tensor(np.zeros(outline.shape[0])).float().to(device)
                for i in range(outline.shape[0]):
                    index = 0

                    for j in range(outline.shape[1]):
                        if outline[i, j] > 0.6:
                            index += 1
                    pre_labels[i] = index
                print("pre_labels", pre_labels)
                print("labels", labels.float())
                loss_line = F.mse_loss(pre_labels, labels.float())
                val_line_loss += loss_line.item()

                pre_labels = torch.tensor(np.zeros(out_branch.shape[0])).float().to(device)

                for i in range(out_branch.shape[0]):
                    index = 0
                    for j in range(out_branch.shape[1]):
                        out_branch[i, j] = torch.softmax(out_branch[i, j], 0)
                        if out_branch[i, j][1] > 0.8:
                            index += 1
                        # index += out_branch[i, j][1]
                    pre_labels[i] = index
                print("pre_labels", pre_labels)
                print("labels", labels.float())
                loss_branch = F.mse_loss(pre_labels, labels.float())
                val_branch_loss += loss_branch.item()

                '''cc calculate '''
                cc_label += labels.tolist()
                cc_predict += logps.tolist()

            print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, val_running_loss / len(val_loader)))

            cc_label = pd.Series(cc_label)
            cc_predict = pd.Series(cc_predict)
            corr = cc_label.corr(cc_predict)
            print("Corr", corr)
            label_predict = pd.concat([cc_label, cc_predict], 1)
            label_predict.to_csv("/correlation_{}.csv".format(epoch), mode='w',
                          index=False, header=True)

            val_loss = val_running_loss / len(val_loader)
            val_branch = val_branch_loss / len(val_loader)
            val_line = val_line_loss / len(val_loader)

            result_list.append([epoch, train_loss, pair_loss, tripet_loss, branch_loss, val_loss,
                                val_branch, val_line, corr])

            name = ['epoch', 'train_loss', 'pair_loss', 'tripet_loss', 'loss_branch', 'val_loss', 'val_branch',
                    'val_line', 'Correlation']
            result = pd.DataFrame(columns=name, data=result_list)
            early_stopping(val_loss, model)

            result.to_csv("adni2_bl.csv", mode='w',
                          index=False, header=True)


    stop = datetime.now()
    print("Running time: ", stop - start)
