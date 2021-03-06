"""
This script provides an example of building a binary neural
network for classifying glass identification dataset on
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
"""

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

"""
Step 1: Load data and pre-process data
"""
# load all data
data = pd.read_csv(r'Anger.csv')
# drop first column as it is identifier
pure_data = data.iloc[1:, 2:]
# try shuffle data
pure_data = pure_data.sample(frac=1).reset_index(drop=True)

# split data into 2 classes to perform binary classification
# by replacing class values in the range of 1-4 (window glass)
# with 1, and 5-7 (non-window glass) with 0
for i in range(pure_data.__len__()):
    # print(pure_data.iloc[i, 6])
    if pure_data.iloc[i, 6] == 'Genuine':
        pure_data.iloc[i, 6] = 1
    else:
        pure_data.iloc[i, 6] = 0

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(pure_data)) < 0.8
train_data = pure_data[msk]
test_data = pure_data[~msk]

n_features = train_data.shape[1] - 1

# split training data into input and target
# the first 9 columns are features, the last one is target
train_input = train_data.iloc[:, :n_features]
train_target = train_data.iloc[:, n_features]

# # normalise training data by columns
for column in train_input:
    train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#     train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# print(train_input)
# print(train_target)

# split training data into input and target
# the first 6 columns are features, the last one is target
test_input = test_data.iloc[:, :n_features]
test_target = test_data.iloc[:, n_features]

# # normalise testing input data by columns
for column in test_input:
    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#     test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())


# create Tensors to hold inputs and outputs
X = torch.Tensor(train_input.values).float()
#print(train_input.values)
train_target = np.array(train_target).astype(np.uint8)
Y = torch.Tensor(train_target).long()

# create Tensors to hold inputs and outputs
X_test = torch.Tensor(test_input.values).float()
test_target = np.array(test_target).astype(np.uint8)
Y_test = torch.Tensor(test_target).long()

"""
Step 2: Define a neural network 
"""

# define the number of inputs, classes, training epochs, and learning rate
input_neurons = n_features
learning_rate = 0.001
num_epochs = 200
output_neurons = 2


# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, layer = 1):
        super(TwoLayerNet, self).__init__()
        self.layer = layer
        if layer == 1:
            self.fc1 = torch.nn.Linear(6, 15)
            self.fc_end = torch.nn.Linear(15, 2)
        elif layer == 2:
            self.fc1 = torch.nn.Linear(6, 15)
            self.fc2_1 = torch.nn.Linear(6, 15)
            self.fc2_2 = torch.nn.Linear(15, 15)
            self.fc_end = torch.nn.Linear(15, 2)
        elif layer ==3:
            self.fc1 = torch.nn.Linear(6, 15)
            self.fc2_1 = torch.nn.Linear(6, 15)
            self.fc2_2 = torch.nn.Linear(15, 15)
            self.fc3_1 = torch.nn.Linear(15, 15)
            self.fc3_2 = torch.nn.Linear(15, 15)
            self.fc_end = torch.nn.Linear(15, 2)
        else:
            raise Exception('too many layers')
    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        if self.layer == 1:
            out = torch.tanh(self.fc1(x))
            y_pred = self.fc_end(out)
            return y_pred
        elif self.layer == 2:
            out1 = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2_1(x))
            out2 = torch.tanh(self.fc2_2(x))
            out = (out1+out2)/2
            y_pred = self.fc_end(out)
            return y_pred
        else:
            out1 = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2_1(x))
            out3 = torch.tanh(self.fc3_1(x))
            out3 = torch.tanh(self.fc3_2(out3))
            out3 = (out3 + x)/2
            out2 = torch.tanh(self.fc2_2(out3))
            out = (out1 + out2)/2
            y_pred = self.fc_end(out)
        return y_pred


# define a neural network using the customised structure
net = TwoLayerNet()

# define loss function
loss_func = torch.nn.CrossEntropyLoss()
# loss_func = torch.nn.MSELoss()

# define optimiser
optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)
# store all losses for visualisation
all_losses = []
all_acc = []
all_acc.append([])
all_acc.append([])
all_acc.append([])
all_acc.append([])
all_acc.append([])
# train a neural network
evolve_time = 4
for evolve in range(1, evolve_time):
    start_time = time.time()
    if evolve == 1:
        net = TwoLayerNet(evolve)
    else:
        pervious_weight = net.state_dict()
        net = TwoLayerNet(evolve)
        new_weight = net.state_dict()
        pretrained_dict = {k: v for k, v in pervious_weight.items() if k in new_weight}
        new_weight.update(pretrained_dict)
        net.load_state_dict(new_weight)
    
    # optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)
    optimiser = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay= 0.01)
    # training ===============================
    for epoch in range(num_epochs + 1):
        # Perform forward pass: compute predicted y by passing x to the model.
        Y_pred = net(X)
        
        # Compute loss
        loss = loss_func(Y_pred, Y)
        all_losses.append(loss.item())
    
        # print progress
        if epoch % 50 == 0:
            # convert three-column predicted Y values to one column for comparison
            print(Y_pred)
            _, predicted = torch.max(Y_pred, 1)
    
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()

            print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

            # test the neural network using testing data
            # It is actually performing a forward pass computation of predicted y
            # by passing x to the model.
            # Here, Y_pred_test contains three columns, where the index of the
            # max column indicates the class of the instance
            Y_pred_test = net(X_test)

            # get prediction
            # convert three-column predicted Y values to one column for comparison
            _, predicted_test = torch.max(Y_pred_test, 1)

            # calculate accuracy
            total_test = predicted_test.size(0)
            correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
            all_acc[int(epoch / 50)].append(100 * correct_test / total_test)
            print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
    
        # Clear the gradients before running the backward pass.
        net.zero_grad()
    
        # Perform backward pass
        loss.backward()
    
        # Calling the step function on an Optimiser makes an update to its
        # parameters
        optimiser.step()

    end_time = time.time()
    print('Time cost: ' + str(end_time - start_time))

    confusion = torch.zeros(output_neurons, output_neurons)

    Y_pred = net(X)

    _, predicted = torch.max(Y_pred, 1)

    for i in range(train_data.shape[0]):
        actual_class = Y.data[i]
        predicted_class = predicted.data[i]

        confusion[actual_class][predicted_class] += 1
    # validation ===============================
    print('')
    print('Confusion matrix for training:')
    print(confusion)
    
    """
    Step 3: Test the neural network
    
    Pass testing data to the built neural network and get its performance
    """
    
    # test the neural network using testing data
    # It is actually performing a forward pass computation of predicted y
    # by passing x to the model.
    # Here, Y_pred_test contains three columns, where the index of the
    # max column indicates the class of the instance
    Y_pred_test = net(X_test)
    
    # get prediction
    # convert three-column predicted Y values to one column for comparison
    _, predicted_test = torch.max(Y_pred_test, 1)
    
    # calculate accuracy
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
    
    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
    
    """
    Evaluating the Results
    
    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every iris flower (rows)
    which class the network guesses (columns).
    
    """
    
    confusion_test = torch.zeros(output_neurons, output_neurons)
    
    for i in range(test_data.shape[0]):
        actual_class = Y_test.data[i]
        predicted_class = predicted_test.data[i]
    
        confusion_test[actual_class][predicted_class] += 1
    
    print('')
    print('Confusion matrix for testing:')
    print(confusion_test)

print(all_acc)

plt.figure()
plt.plot(all_acc)
plt.show()
plt.plot(all_losses)
plt.show()
# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss



