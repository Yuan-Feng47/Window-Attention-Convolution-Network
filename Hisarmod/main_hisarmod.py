import copy
import os

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torch.utils.tensorboard.writer
import shutil

from Net.WACN import wacn_b0

# from fix import fix_state_dict
from HisarMod_filter_classes_str_dic import convert_labels_to_dict, tensor_to_labels
from getdataset_HisarMod import ModulationRecognitionDataset_HisarMod

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.metrics import accuracy_score
import random


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

print(torch.cuda.is_available())

# 检查可用的 GPU 数量
num_gpus = torch.cuda.device_count()
print("Number of available GPUs: ", num_gpus)
# 使用所有可用的 GPU
devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
print("Using devices:", devices)

# 设置参数
num_epochs = 1000
learning_rate = 0.0002
batch_size = 400
patience = 8
# 数据集预加载  'RML2018.01a'  'HisarMod2019.01'
dataset_type = 'HisarMod2019.01'

if dataset_type =='HisarMod2019.01':
    print('HisarMod2019.01')
    select_every = 1
    # 建议为500的约数
    train_name = './train100.hdf5'
    test_name = './test100.hdf5'

    snr_list = list(range(-20, 18 + 1, 2))
    HisarMod_list = list(range(26))

    classes_dic = convert_labels_to_dict(HisarMod_list)
    print('classes_dic:', classes_dic)
    num_classes = len(classes_dic)
    train_dataset = ModulationRecognitionDataset_HisarMod(train_name, select_every, snr_list, HisarMod_list)
    valid_dataset = ModulationRecognitionDataset_HisarMod(test_name, select_every, snr_list, HisarMod_list)

# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
# test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
num_classes = 26
# 初始化模型、损失函数和优化器
net = wacn_b0(num_classes=num_classes)
# 存文件的名称准备
network_name = type(net).__name__  # 获取网络名称

if num_gpus > 1:
    net = nn.DataParallel(net, device_ids=devices)
net = net.to(devices[0])  # 将主模型移到第一个 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_model_state = copy.deepcopy(net.state_dict())
# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3, verbose=True)

# Early Stopping变量
best_val_loss = np.inf
epochs_since_improvement = 0
best_val_accuracy = 0

# 开始时间用于计算总耗时
start_time = time.time()

if os.path.exists("./log"):
    assert os.path.isdir("./log")
    shutil.rmtree("./log")

log_writer = torch.utils.tensorboard.writer.SummaryWriter("./log")

# 训练循环
for epoch in range(num_epochs):
    net.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    with tqdm(enumerate(train_loader), total=len(train_loader), unit='batch') as train_progress_bar:
        for batch_idx, (data, target, snr) in train_progress_bar:
            data = data.to(devices[0])  # 将数据移动到 GPU（如果有）
            target = target.to(devices[0])  # 将标签移动到 GPU（如果有）
            optimizer.zero_grad()
            output = net(data)

            if dataset_type == 'HisarMod2019.01'or'备份':
                target_labels = tensor_to_labels(target).to(devices[0])  # 转换为类别标签
                # 转换arget_labels = tensor_to_labels(target).to(devices[0])为类别标签

            loss = criterion(output, target_labels)
            loss.backward()
            optimizer.step()

            # 计算训练精度
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target_labels).sum().item()
            train_total += target_labels.size(0)

            # 更新进度条
            train_progress_bar.set_description('Epoch {}/{}: Training'.format(epoch + 1, num_epochs))
            train_progress_bar.set_postfix(loss=loss.item())

    train_accuracy = 100.0 * train_correct / train_total
    train_loss /= len(train_loader)

    # 验证

    net.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(enumerate(valid_loader), total=len(valid_loader), unit='batch',
                  desc='Validating') as val_progress_bar:
            for batch_idx, (data, target, snr) in val_progress_bar:
                data = data.to(devices[0])
                target = target.to(devices[0])

                output = net(data)

                if dataset_type == 'HisarMod2019.01'or'备份':
                    target_labels = tensor_to_labels(target).to(devices[0])  # 转换为类别标签

                loss = criterion(output, target_labels)

                val_loss += loss.item()
                probabilities = torch.softmax(output, dim=1)  # 计算类别概率
                max_probs, predicted = torch.max(probabilities.data, 1)

                val_correct += (predicted == target_labels).sum().item()
                val_total += target_labels.size(0)

                # 更新进度条
                val_progress_bar.set_postfix(loss=loss.item())

    val_accuracy = 100.0 * val_correct / val_total
    val_loss /= len(valid_loader)
    # 更新学习率
    scheduler.step(val_loss)
    print("Epoch {}/{} - Train Loss: {:.9f}, Accuracy: {:.7f}% - Validation Loss: {:.9f}, Accuracy: {:.7f}%".format(
        epoch + 1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_since_improvement = 0
        best_model_state = copy.deepcopy(net.state_dict())
        # torch.save(net.state_dict(), "model_Done.pth")
        print("Validation accuracy improved. Saving the current model weights.")

    print('esi={}'.format(epochs_since_improvement))


    if epochs_since_improvement == patience:
        print("Early stopping. Training stopped after {} epochs.".format(epoch + 1))
        print("best_val_accuracy = {}.".format(best_val_accuracy))
        end_time = time.time()
        print('程序运行时间为: %s Seconds' % (end_time - start_time))
        break
    # 检查当前验证集准确率是否是最大的，如果是，则更新保存的权重

# 预定义多信噪比下的准确度
test_accuracies = []
# 准备空预测和真实标签列表
y_true_all = []
y_pred_all = []

# 加载最佳模型
net.eval()
net.load_state_dict(best_model_state)
for snr_show in snr_list:
    # 创建一个与 snr_show 对应的 Tensor
    snr_show_tensor = torch.tensor([snr_show]).to('cpu')
    print('正在验证SNR={}时的识别准确性'.format(snr_show))

    # 准备空预测和真实标签列表
    y_true = []
    y_pred = []

    # 测试集上进行预测
    with torch.no_grad():
        for data, target, snr in tqdm(valid_loader):
            snr_tensor = snr.clone().detach()
            # 找到所有SNR等于snr_show的样本的索引
            indices = (snr_tensor == snr_show_tensor).nonzero(as_tuple=True)[0]
            # 使用索引过滤数据
            data_test = data[indices]
            target_test = target[indices]

            if data_test.size(0) == 0:  # 如果这个批次没有数据，就跳过
                continue
            data_test = data_test.to(devices[0])  # 将数据移动到第一个 GPU（如果有）

            output = net(data_test)

            if dataset_type == 'HisarMod2019.01'or'备份':
                target_labels = tensor_to_labels(target_test).to(devices[0])  # 转换为类别标签

            probabilities = torch.softmax(output, dim=1)  # 计算类别概率val
            max_probs, predicted_labels = torch.max(probabilities.data, 1)

            y_true.extend(target_labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())  # 从 GPU 移到 CPU 以便与 numpy 一起使用

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"SNR: {snr_show}, test_accuracy: {test_accuracy}")
    test_accuracies.append(test_accuracy)
    # 存储画总体混淆矩阵的
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)
    # 归一化混淆矩阵
    row_sums = cm.sum(axis=1)
    for i, row_sum in enumerate(row_sums):
        if row_sum == 0:
            cm[i, i] = 1
            row_sums[i] = 1
    normalized_cm = cm.astype('float') / row_sums[:, np.newaxis]

    # 绘制混淆矩阵
    matplotlib.use("Agg")
    plt.figure(figsize=(16, 16))
    plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes_dic))
    plt.xticks(tick_marks, classes_dic.values(), rotation=90)
    plt.yticks(tick_marks, classes_dic.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix:')

    # 在每个格子中显示数量
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            plt.text(j, i, format(normalized_cm[i, j], ".2f"), horizontalalignment="center",
                     color="white" if normalized_cm[i, j] > 0.5 else "black")

    # 保存混淆矩阵为PNG格式
    pic_filename = f"{network_name}+{dataset_type}_snr_{snr_show}_test_acc_{test_accuracy:.4f}.png"
    plt.savefig(pic_filename, bbox_inches='tight')
    plt.close()
print("测试结束！！！！！！！！！！！！")

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(snr_list, test_accuracies, 'o-')
plt.xlabel('SNR')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs SNR')
# 获取当前坐标轴
ax = plt.gca()
# 设置 x 轴和 y 轴的主刻度值
ax.set_xticks(np.arange(min(snr_list), max(snr_list), 2))  # 这里假设 snr_list 是 SNR 的数值列表
ax.set_yticks(np.arange(0, 1.1, 0.1))  # 假设 Test Accuracy 的取值范围在 [0, 1] 之间
# 设置网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('{}+{}.png'.format(network_name, dataset_type), dpi=300, bbox_inches='tight')
plt.close()
# 计算总混淆矩阵
cm_all = confusion_matrix(y_true_all, y_pred_all)
test_accuracy_all = accuracy_score(y_true_all, y_pred_all)
print(f"Overall test_accuracy: {test_accuracy_all}")

# 归一化混淆矩阵
row_sums_all = cm_all.sum(axis=1)
for i, row_sum in enumerate(row_sums_all):
    if row_sum == 0:
        cm_all[i, i] = 1
        row_sums_all[i] = 1
normalized_cm_all = cm_all.astype('float') / row_sums_all[:, np.newaxis]

# 绘制混淆矩阵
plt.figure(figsize=(16, 16))
plt.imshow(normalized_cm_all, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes_dic))
plt.xticks(tick_marks, classes_dic.values(), rotation=90)
plt.yticks(tick_marks, classes_dic.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix:')

# 在每个格子中显示数量
for i in range(normalized_cm_all.shape[0]):
    for j in range(normalized_cm_all.shape[1]):
        plt.text(j, i, format(normalized_cm_all[i, j], ".2f"), horizontalalignment="center",
                 color="white" if normalized_cm_all[i, j] > 0.5 else "black")

# 保存混淆矩阵为PNG格式
pic_filename_all = f"{network_name}+{dataset_type}_overall_test_acc_{test_accuracy_all:.4f}.png"
plt.savefig(pic_filename_all, bbox_inches='tight')
plt.close()

