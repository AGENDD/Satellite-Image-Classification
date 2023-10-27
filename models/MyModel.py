import numpy
import torch
from torch import nn
import torch.optim as optim

class Modell(nn.Module):


    def __init__(self):
        super(Modell, self).__init__()

        self.name = "Model-1"

        self.features = nn.Sequential(
            #添加第一层卷积层，有32个输出（即32个卷积核），卷积核大小为3x3，输入形状为255x255的彩色图像（3通道），激活函数为ReLU。
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            #添加第二层卷积层，有32个输出，卷积核大小为3x3，输入形状为64x64的彩色图像（3通道），激活函数为ReLU。
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            #添加第一个最大池化层，池化窗口大小为2x2。
            nn.MaxPool2d(2, 2),
            #添加第三层卷积层，有64个输出，卷积核大小为3x3，激活函数为ReLU。
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            #添加第二个最大池化层，池化窗口大小为2x2。
            nn.MaxPool2d(2, 2),
            #添加第四层卷积层，有128个输出，卷积核大小为3x3，激活函数为ReLU。
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            #添加第三个最大池化层，池化窗口大小为2x2。
            nn.MaxPool2d(2, 2),
        )

        # #添加第一层卷积层，有32个输出（即32个卷积核），卷积核大小为3x3，输入形状为255x255的彩色图像（3通道），激活函数为ReLU。
        # self.conv1 = nn.Conv2d(3, 32, 3)
        # self.relu1 = nn.ReLU()
        # #添加第二层卷积层，有32个输出，卷积核大小为3x3，输入形状为64x64的彩色图像（3通道），激活函数为ReLU。
        # self.conv2 = nn.Conv2d(32, 32, 3)
        # self.relu2 = nn.ReLU()
        # #添加第一个最大池化层，池化窗口大小为2x2。
        # self.pool1 = nn.MaxPool2d(2, 2)
        # #添加第三层卷积层，有64个输出，卷积核大小为3x3，激活函数为ReLU。
        # self.conv3 = nn.Conv2d(32, 64, 3)
        # self.relu3 = nn.ReLU()
        # #添加第二个最大池化层，池化窗口大小为2x2。
        # self.pool2 = nn.MaxPool2d(2, 2)
        # #添加第四层卷积层，有128个输出，卷积核大小为3x3，激活函数为ReLU。
        # self.conv4 = nn.Conv2d(64, 128, 3)
        # self.relu4 = nn.ReLU()
        # #添加第三个最大池化层，池化窗口大小为2x2。
        # self.pool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            #添加全连接层（即密集连接层），有128个输出节点，激活函数为ReLU。
            nn.Linear(128 * 6 * 6, 128), # 这里的26*26取决于输入图像的大小和卷积核的大小
            nn.ReLU(),
            #添加Dropout层，丢弃率为0.5。Dropout可以防止模型过拟合。
            nn.Dropout(0.5),
            #添加输出层（全连接层），有4个节点对应4个类别，激活函数为softmax。softmax可以将输出转化为概率分布。
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )

        # #对前一层的输出进行扁平化。
        # #添加全连接层（即密集连接层），有128个输出节点，激活函数为ReLU。
        # self.fc1 = nn.Linear(128 * 26 * 26, 128) # 这里的26*26取决于输入图像的大小和卷积核的大小
        # self.relu5 = nn.ReLU()
        # #添加Dropout层，丢弃率为0.5。Dropout可以防止模型过拟合。
        # self.dropout1 = nn.Dropout(0.5)
        # #添加输出层（全连接层），有4个节点对应4个类别，激活函数为softmax。softmax可以将输出转化为概率分布。
        # self.fc2 = nn.Linear(128, 4)
        # self.softmax1 = nn.Softmax(dim=1)

    
    def forward(self, x):

        x = self.features(x)
        #x = torch.flatten(x, 1)
        # print(x.size())
        x = x.view(x.shape[0], -1)
        # print(x.size())
        output = self.classifier(x)
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.pool2(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.pool3(x)
        # x = torch.flatten(x, 1) # 展平操作
        # x = self.fc1(x)
        # x = self.relu5(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # output = self.softmax1(x)
        
        return output
