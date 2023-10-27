import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import random
import copy

import models.MyModel as MyModel#这里切换模型

import shutil
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
# import matplotlib as plt




###################################################################  初始化
print("Initializing...")
num_epochs = 20
SEED = 114514
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
mean = [0.40011665, 0.43133965, 0.4274904]
std = [0.02309906, 0.020635523, 0.018958775]

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device('cuda')
else:
    print("GPU is not available.")
    device = torch.device('cpu')


data_path = os.path.join(os.getcwd(), "data")
split_data_path = os.path.join(os.getcwd(), "split_data")

true_labels = []

# 遍历"data"目录下的所有文件（或文件夹），每个文件（或文件夹）名被视为一个标签
for label in os.listdir(data_path):
    # 将每个标签的完整路径和标签名作为一个列表添加到labels列表中
    true_labels.append([os.path.join(data_path, label), label])
   

# labels为列表数组，每个列表元素存有对应图片的文件夹路径以及label
# 访问的时候可以用二维数组下标进行访问



############################################################################    预处理
print("Preprocessing...")
train_ratio = 0.8
test_ratio = 0.2
valid_ratio = 0.9
# 对训练集/测试集进行预处理
train_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomRotation(45),                   #随机旋转图像，旋转角度在-45到45度
    transforms.RandomHorizontalFlip(0.5),           #以50%的概率水平翻转图像。
    transforms.RandomCrop(64, padding=4),           #对图像进行填充（扩大图像），然后再随机裁剪出64x64大小的图像。填充大小为4。
    transforms.ToTensor(),                          #将PIL Image或者numpy.ndarray数据类型的图像转换为torch.Tensor，并且会自动将图像的数据范围调整到[0, 1]。
    transforms.Normalize(mean=mean, std = std)
    ])

test_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])

image_data = datasets.ImageFolder(data_path)

# image_data_test_transforms = datasets.ImageFolder(data_path)
train_size = int(len(image_data) * train_ratio)
test_size = len(image_data) - train_size

#获得training data
train_data, test_data = data.random_split(image_data, [train_size, test_size])
# train_data = copy.deepcopy(train_data)
# test_data = copy.deepcopy(test_data)
# test_data.dataset.transform = test_transforms
# print(type(test_data))
# print((test_data[0][1]))
# test_data[0][0].show()
n_train_size = int(train_size * valid_ratio)
n_valid_size = train_size - n_train_size
train_data, valid_data = data.random_split(train_data, [n_train_size, n_valid_size])

# train_data = copy.deepcopy(train_data)
# valid_data = copy.deepcopy(valid_data)
# valid_data.dataset.transform = test_transforms
# print(type(valid_data))
# print(type(valid_data[0][0]))

# train_data.dataset.transform = train_transforms
# print(type(train_data))
# print(type(train_data[0][0]))
# 对不同集设置对应的transform的方式
print("Saving split data to local work directories")
if os.path.join(os.getcwd(), "split_data", "train"):
    shutil.rmtree(os.path.join(os.getcwd(), "split_data", "train"), ignore_errors=True)
if os.path.join(os.getcwd(), "split_data", "test"):
    shutil.rmtree(os.path.join(os.getcwd(), "split_data", "test"), ignore_errors=True)
if os.path.join(os.getcwd(), "split_data", "valid"):
    shutil.rmtree(os.path.join(os.getcwd(), "split_data", "valid"),ignore_errors=True)

for i in range(0, len(true_labels)):
    os.makedirs(os.path.join(os.getcwd(), "split_data", "train", true_labels[i][1]))
    os.makedirs(os.path.join(os.getcwd(), "split_data", "test", true_labels[i][1]))
    os.makedirs(os.path.join(os.getcwd(), "split_data", "valid", true_labels[i][1]))



for i in range(0, len(train_data)):
    img = train_data[i][0]
    # print(type(img))
    img.save(os.path.join(os.getcwd(), "split_data", "train", true_labels[train_data[i][1]][1], (str(i)+".jpg")))

for i in range(0, len(test_data)):
    img = test_data[i][0]
    img.save(os.path.join(os.getcwd(), "split_data", "test", true_labels[test_data[i][1]][1], (str(i)+".jpg")))

for i in range(0, len(valid_data)):
    img = valid_data[i][0]
    img.save(os.path.join(os.getcwd(), "split_data", "valid", true_labels[valid_data[i][1]][1], (str(i)+".jpg")))

train_data = datasets.ImageFolder(os.path.join(split_data_path, "train"), transform=train_transforms)

test_data = datasets.ImageFolder(os.path.join(split_data_path, "test"), transform=test_transforms)
valid_data = datasets.ImageFolder(os.path.join(split_data_path, "valid"), transform=test_transforms)
# valid_data 是 training data 的子集， 需要使用deepcopy防止影响到training data
# valid_data = copy.deepcopy(valid_data)
# valid_data.dataset.transform = test_transforms
print("Size of training data:\t", len(train_data))
print("Size of test data:\t", len(test_data))
print("Size of valid data:\t", len(valid_data))
# print(train_data.class_to_idx)
# print(test_data.class_to_idx)
# print(valid_data.class_to_idx)

# print(len(train_data), len(test_data), len(valid_data))


#将训练数据加载到一个数据加载器（DataLoader）中
BATCH_SIZE = 128
# train_loader = data.DataLoader()
train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

print(train_iterator)

x_axis = []
acc = []
for i in range(1,num_epochs+1):
    x_axis.append(i)

##############################################################################   建模
print("Building model...")
model = MyModel.Modell()

# 将你的网络模型移动到相应的设备上
model = model.to(device)
# 定义损失函数，使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 选择优化器，使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)




##############################################################################   训练
print("Training...")

for epoch in range(num_epochs):  # num_epochs 是总迭代次数
    for i, (inputs, labels) in enumerate(train_iterator):  # 遍历数据加载器以获取输入和标签

        inputs = inputs.to(device)  # 将输入数据移动到正确的设备上
        labels = labels.to(device)  # 如果目标也是张量，也需要移动到正确的设备上


        # 前向传播 
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 优化步骤
        optimizer.step()
        # 清零梯度
        optimizer.zero_grad()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        # 在每个epoch结束后，使用验证数据集来评估模型
    model.eval()  # 设置模型为评估模式


    total_loss = 0
    total_correct = 0
    total_samples = 0


    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in valid_iterator:  # 遍历验证数据加载器

            inputs = inputs.to(device)  # 将输入数据移动到正确的设备上
            labels = labels.to(device)  # 如果目标也是张量，也需要移动到正确的设备上         


            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            
            total_correct += (preds == labels).sum().item()
            total_loss += val_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)


    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    print(f'Validation Loss: {avg_loss}, Accuracy: {avg_accuracy}')

    acc.append(avg_accuracy)
    model.train()  # 将模型设置回训练模式



############################################################################    测试
print("Testing...")
# 设置模型为评估模式
model.eval()

# 初始化统计变量
correct = 0
total = 0

# 不需要计算梯度
with torch.no_grad():
    for inputs, labels in test_iterator:  # 遍历测试数据加载器

        inputs = inputs.to(device)  # 将输入数据移动到正确的设备上
        labels = labels.to(device)  # 如果目标也是张量，也需要移动到正确的设备上


        # 前向传播
        outputs = model(inputs)
        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 更新统计变量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
accuracy = 100 * correct / total

print(f'Accuracy of the model on the test data: {accuracy}%')

plt.plot(x_axis, acc)
plt.show()
plt.savefig(f"figs/{model.name}.png")


print("Saving model...")
torch.save(model, f'model_outputs/{model.name}.pth')
print("Finish")


