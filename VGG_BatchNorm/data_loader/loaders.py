# data/loaders.py
import os
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        if n_items < 0 or n_items > len(dataset):  # 处理 n_items 为 -1 或超出范围的情况
            self.n_items = len(dataset)
        else:
            self.n_items = n_items

    def __getitem__(self, index):
        if index >= self.n_items:  # 确保索引在部分数据集的范围内
            raise IndexError(f"索引 {index} 超出 PartialDataset (大小: {self.n_items}) 的范围")
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.n_items


def get_cifar_loader(root='../../data/', batch_size=128, train=True, shuffle=True, num_workers=1, n_items=-1):
    # CIFAR-10 的标准均值和标准差
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # 若目的为训练，则对数据启用数据增强
    if train:
        data_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            transforms.ToTensor(),
            normalize
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # torchvision 会自动检查 root 目录下载数据
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    
    if n_items != -1:  # 仅当 n_items 不是请求完整数据集时才包装
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return loader


if __name__ == '__main__':
    test_data_dir = '../../data_cifar'
    os.makedirs(test_data_dir, exist_ok=True)
    
    print("测试 CIFAR-10 加载器...")
    # 使用少量数据进行快速测试
    train_loader_sample = get_cifar_loader(root=test_data_dir, n_items=10, batch_size=4) 
    print(f"训练加载器中的批次数: {len(train_loader_sample)}")

    if len(train_loader_sample) > 0:
        for X, y in train_loader_sample:
            print("示例批次已加载:")
            # X shape: torch.Size([4, 3, 32, 32]), y shape: torch.Size([4])
            print(f"X 形状: {X.shape}, y 形状: {y.shape}")
            print(f"X[0] 样本 (第一个通道的前3个值):\n{X[0, 0, 0, :3]}")
            print(f"y[0] 样本: {y[0]}")
            
            # 可视化 (需要先反归一化)
            img = X[0].numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean  # 反归一化
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.savefig('sample_cifar.png')
            print("已保存 sample_cifar.png")
            break
    else:
        print("训练加载器为空或 n_items 对于一个批次来说太小。")
    
    # 清理临时目录
    '''import shutil
    shutil.rmtree(test_data_dir)
    print(f"已清理 {test_data_dir}")'''
