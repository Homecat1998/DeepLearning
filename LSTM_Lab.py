import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import xlrd
from numpy import *
import random
import math

# all the names of worksheets
# hardcoded as it is always fixed for our data
# other inits
worksheet_names = list(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10',
                        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])
all_left_true = list()
all_right_true = list()
all_left_false = list()
all_right_false = list()
all_pairs_true = list()
all_pairs_false = list()
print(worksheet_names)

# first open the raw data for left eye
# go through the worksheets
# read col by col, pointing out True or False
with xlrd.open_workbook('PDleft.xlsx') as workbook_left:
    for sheet_name in worksheet_names:
        worksheet = workbook_left.sheet_by_name(sheet_name)
        for col_index in range(worksheet.ncols):
            if worksheet.cell(1, col_index).value == '':
                continue
            new_column = list()

            for row_index in range(1, worksheet.nrows):
                new_column.append(float(worksheet.cell(row_index, col_index).value))

            # if the data is True Data
            if sheet_name.__contains__('T'):
                all_left_true.append(new_column)
            else:
                all_left_false.append(new_column)

# the same job for the right one
# pointing out true or false
with xlrd.open_workbook('PDright.xlsx') as workbook_right:
    for sheet_name in worksheet_names:
        worksheet = workbook_right.sheet_by_name(sheet_name)
        for col_index in range(worksheet.ncols):
            if worksheet.cell(1, col_index).value == '':
                continue
            new_column = list()
            for row_index in range(1, worksheet.nrows):
                new_column.append(float(worksheet.cell(row_index, col_index).value))

            if sheet_name.__contains__('T'):
                all_right_true.append(new_column)
            else:
                all_right_false.append(new_column)

true_samples = all_right_true.__len__()
false_samples = all_right_false.__len__()

# process true samples
for index in range(true_samples):
    origin_left = all_left_true.__getitem__(index)
    origin_right = all_right_true.__getitem__(index)
    no_zero_left = list()
    no_zero_right = list()
    processed_left = list()
    processed_right = list()
    rtn = list()

    # only when both left and right eye is not 0
    # we will use this entry
    # otherwise skip it
    for index_inner in range(origin_left.__len__()):
        if origin_left.__getitem__(index_inner) != 0.0 and origin_right.__getitem__(index_inner) != 0.0:
            no_zero_left.append(origin_left.__getitem__(index_inner))
            no_zero_right.append(origin_right.__getitem__(index_inner))

    # get the mean and Variance
    left_mean = mean(no_zero_left)
    right_mean = mean(no_zero_right)
    left_var = var(no_zero_left)
    right_var = var(no_zero_right)

    # standardize the data
    # by (data - mean) / var
    for index_process in range(no_zero_left.__len__()):
        processed_left.append((no_zero_left.__getitem__(index_process) - left_mean) / left_var)
        processed_right.append((no_zero_right.__getitem__(index_process) - right_mean) / right_var)

    final_left = list()
    final_right = list()

    # Resampling, make sure all data has the length of 100
    to_throw = -1;
    if no_zero_left.__len__() >= 100:
        to_throw = no_zero_left.__len__() - 100

        if to_throw != 0:
            interval = math.ceil(no_zero_left.__len__() / to_throw)
            for index_inner_2 in range(no_zero_left.__len__()):
                if (index_inner_2 + 1) != interval:
                    final_left.append(processed_left.__getitem__(index_inner_2))
                    final_right.append(processed_right.__getitem__(index_inner_2))

                if final_left.__len__() == 100:
                    break

    # only add valid data to the list
    # that we will use later
    if to_throw != -1 and final_left.__len__() > 0:
        rtn.append(final_left)
        rtn.append(final_right)
        rtn.append(1)
        all_pairs_true.append(rtn)

# the same job for false samples
# the core part is the same as above
for index in range(false_samples):
    origin_left = all_left_false.__getitem__(index)
    origin_right = all_right_false.__getitem__(index)
    no_zero_left = list()
    no_zero_right = list()
    processed_left = list()
    processed_right = list()
    rtn = list()

    for index_inner in range(origin_left.__len__()):
        if origin_left.__getitem__(index_inner) != 0.0 and origin_right.__getitem__(index_inner) != 0.0:
            no_zero_left.append(origin_left.__getitem__(index_inner))
            no_zero_right.append(origin_right.__getitem__(index_inner))

    left_mean = mean(no_zero_left)
    right_mean = mean(no_zero_right)
    left_var = var(no_zero_left)
    right_var = var(no_zero_right)

    for index_process in range(no_zero_left.__len__()):
        processed_left.append((no_zero_left.__getitem__(index_process) - left_mean) / left_var)
        processed_right.append((no_zero_right.__getitem__(index_process) - right_mean) / right_var)

    final_left = list()
    final_right = list()

    to_throw = -1
    if no_zero_left.__len__() > 100:
        to_throw = no_zero_left.__len__() - 100

        if to_throw != 0:
            interval = math.ceil(no_zero_left.__len__() / to_throw)
            for index_inner_2 in range(no_zero_left.__len__()):
                if (index_inner_2 + 1) != interval:
                    final_left.append(processed_left.__getitem__(index_inner_2))
                    final_right.append(processed_right.__getitem__(index_inner_2))

                if final_left.__len__() == 100:
                    break

    if to_throw != -1 and final_left.__len__() > 0:
        rtn.append(final_left)
        rtn.append(final_right)
        rtn.append(0)

        all_pairs_false.append(rtn)

print('Total ' + str(all_pairs_true.__len__()) + ' pair of true data')
print('Total ' + str(all_pairs_false.__len__()) + ' pair of false data')


# set up the network
# the first layer is lstm
# then a 64-32 full connect linear
# finally a 32-2 full connect linear one for output
# activation between linear layers is relu
class simpleLSTM(nn.Module):
    def __init__(self):
        super(simpleLSTM, self).__init__()
        self.LSTM = nn.LSTM(2, 2, 2)
        self.FC = nn.Linear(32, 16)
        self.FC2 = nn.Linear(16, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hprev, cprev):
        output, hc = self.LSTM(input.float(), (hprev, cprev))
        # retrieve the last 16 lines of the 'stream'
        # and flatten them to fit in the linear network
        output = output[-16:, :, :]
        output = output.view(1, 1, 32)
        output = self.FC(output)
        output = self.relu(output)
        output = self.FC2(output)
        return output, hc
        # return output.view(seq_len, len(chars)), hc


learning_rate = 0.01
net = simpleLSTM()
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
num_epochs = 300

# init h0 and c0
hprev = torch.randn(2, 1, 2).float()
cprev = torch.randn(2, 1, 2).float()

for k in range(5):

    # shuffle the input
    random.shuffle(all_pairs_true)
    random.shuffle(all_pairs_false)

    # split the training data
    # and testing data
    # as k = 5, so each time is 1/5 = 0.2
    # the training part is 1 - 0.2 = 0.8
    print('Currently training fold: ' + str(k))
    true_split = int(all_pairs_true.__len__() * 0.8)
    false_split = int(all_pairs_false.__len__() * 0.8)
    print('True data split point: ' + true_split.__str__())
    print('False data split point: ' + false_split.__str__())

    train_data_total = all_pairs_true[0:true_split]
    train_data_total = train_data_total + all_pairs_false[0:false_split]
    test_data_total = all_pairs_true[true_split:]
    test_data_total = test_data_total + all_pairs_false[false_split:]

    for epoch in range(num_epochs):

        print("Epoch: " + str(epoch))
        epoch_correct = 0
        err = 0

        # shuffle the data for every single epoch
        random.shuffle(train_data_total)

        for train_index in range(train_data_total.__len__()):
            train_data = train_data_total.__getitem__(train_index)
            seq_len = train_data.__getitem__(0).__len__()
            train_input_left = torch.tensor([train_data.__getitem__(0)])
            train_input_right = torch.tensor([train_data.__getitem__(1)])

            # format the training data
            data_input = torch.cat((train_input_left, train_input_right), 0)
            data_input = data_input.view(seq_len, 1, 2)

            target = train_data.__getitem__(-1)
            if target == 1:
                target = torch.tensor([1])
            else:
                target = torch.tensor([0])

            net.zero_grad()
            output, hc = net(data_input, hprev, cprev)
            output = output[-1]

            if output[0][0] < output[0][1]:
                predicted = 1
            else:
                predicted = 0

            if predicted == train_data.__getitem__(-1):
                epoch_correct = epoch_correct + 1
            hprev = hc[0].detach()
            cprev = hc[1].detach()
            err = loss_func(output, target)

            err.backward()
            optimiser.step()

        print("Accuracy on training: ")
        print(str(100 * epoch_correct / train_data_total.__len__()) + '%')

    # testing part for every single fold.
    test_correct = 0
    random.shuffle(test_data_total)
    for test_index in range(test_data_total.__len__()):
        test_data = test_data_total.__getitem__(test_index)
        seq_len = test_data.__getitem__(0).__len__()
        test_input_left = torch.tensor([test_data.__getitem__(0)])
        test_input_right = torch.tensor([test_data.__getitem__(1)])
        data_input = torch.cat((test_input_left, test_input_right), 0)
        data_input = data_input.view(seq_len, 1, 2)

        target = test_data.__getitem__(-1)
        if target == 1:
            target = torch.tensor([1])
        else:
            target = torch.tensor([0])

        output, hc = net(data_input, hprev, cprev)
        output = output[-1]

        if output[0][0] < output[0][1]:
            predicted = 1
        else:
            predicted = 0

        if predicted == test_data.__getitem__(-1):
            test_correct = test_correct + 1

    print("Accuracy on testing: ")
    print(str(100 * test_correct / test_data_total.__len__()) + '%')
