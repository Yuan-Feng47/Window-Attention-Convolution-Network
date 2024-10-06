import torch
import torch.nn as nn


class CNN2Model(nn.Module):
    def __init__(self, input_shape=[2, 128], num_classes=26):
        super(CNN2Model, self).__init__()
        self.conv1 = nn.Conv1d(2, 256, kernel_size= 8, padding='same')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, 128, kernel_size= 8, padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size= 2,stride = 2)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(128, 64, kernel_size= 8, padding='same')
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size= 2,stride = 2)
        self.dropout3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(64, 64, kernel_size= 8, padding='same')
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size= 2,stride = 2)
        self.dropout4 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        flattened_shape = self._get_flattened_shape(input_shape)
        self.dense1 = nn.Linear(flattened_shape, 128)

        # self.dense1 = nn.Linear(64 * (input_shape[1] // 16), 128)
        self.relu5 = nn.ReLU()
        self.dense2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

    def _get_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        flattened_shape = x.size(1) * x.size(2)
        return flattened_shape