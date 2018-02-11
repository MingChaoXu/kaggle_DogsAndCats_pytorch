import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        :param root: 图片地址
        :param transforms: 数据增强
        :param train: 用作训练集
        :param test: 用作测试集
        '''
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test: data/test/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split['/'][-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练、验证集，验证：训练 = 3：7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            # 数据转换操作， 测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = transforms
            pass

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        :param index: 图片索引
        :return: 图片数据和标签
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        :return:数据集中所有图片的个数
        '''
        return len(self.imgs)
