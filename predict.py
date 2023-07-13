import torch
import os
from utils import get_yaml_value, parameter, create_dir, save_feature_network
from torchvision import datasets, transforms

classes = get_yaml_value("classes")
batchsize = get_yaml_value("batch_size")

data_dir = get_yaml_value("dataset_path")
image_size = get_yaml_value("image_size")

name_rank = []
with open("query_drone_name.txt", "r") as f:
    for txt in f.readlines():
        name_rank.append(txt[:-1])

def rotate(img, k):
    """
    Rotate image

    Parameters:
    img (torch.Tensor): the input image tensor
    k (int): the number of times the image is rotated by 90 degrees

    Returns:
    img_rotated (torch.Tensor): the rotated image tensor
    """
    img_rotated = torch.rot90(img, k, [2, 3])
    return img_rotated

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

# print(gallery_img_list)
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, file_list, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._make_dataset(file_list)
        # print(self.samples)

    def _make_dataset(self, file_list):
        data = []
        for line in file_list:
            path = os.path.join(self.root,"query_drone_160k" ,line)
            item = (path, int(0))
            data.append(item)
        return data

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

transform_test_list = [
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

data_transforms = {
    'drone': transforms.Compose(transform_test_list),
    'satellite': transforms.Compose(transform_test_list)}
image_datasets = {}

image_datasets['satellite'] = datasets.ImageFolder(os.path.join("/root", 'autodl-tmp', 'Gallery'),
                                                   data_transforms['satellite'])
image_datasets['drone'] = CustomImageFolder(os.path.join("/root", 'autodl-tmp', 'Query'),name_rank,
                                               data_transforms['drone'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=3,
                                              shuffle=False, num_workers=8, pin_memory=True)              
               for x in ['satellite', 'drone']}
with open('query_drone_name.txt', 'r') as f:
    order = [line.strip() for line in f.readlines()]
image_datasets['drone'].imgs = sorted(image_datasets['drone'].imgs, key=lambda x: order.index(x[0].split("/")[-1]))


import torch
from torch import nn
import model_
net_path = "/root/autodl-tmp/weights/Modern_1652_2023-06-28-22:10:01/net_054.pth"
model = model_.EVA(1402, 0.1).cuda()
model.load_state_dict(torch.load(net_path))

model = model.eval()

device = "cuda:0"
import numpy as np
import time
def extract_feature(model, dataloaders, block, LPN, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n

        if LPN:
            ff = torch.FloatTensor(n, 512, block).zero_().cuda()
        else:
            ff = torch.FloatTensor(n, 512).zero_().cuda()

        # why for in range(2)：
        # 1. for flip img
        # 2. for normal img

        for i in range(2):
            if i == 1:
                img = fliplr(img)
            # elif i == 3:
            #     img = rotate(img, 2)


            input_img = img.to(device)
            outputs = None
            since = time.time()

            if view_index == 1:
                outputs, _ = model(input_img, None)
            elif view_index == 2:
                _, outputs = model(None, input_img)
            # print(outputs.shape)
            # print(ff.shape)
            ff += outputs
            time_elapsed = time.time() - since
            # print(time_elapsed)
            # ff.shape = [16, 512, 4]

        if LPN:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block)
            # print("fnorm", fnorm.shape)
            ff = ff.div(fnorm.expand_as(ff))
            # print("ff", ff.shape)
            ff = ff.view(ff.size(0), -1)
            # print("ff", ff.shape)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # print("fnorm", fnorm.shape)
            ff = ff.div(fnorm.expand_as(ff))
            # print("ff", ff.shape)

        features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
    return features

print("extract")
query_feature = extract_feature(model, dataloaders["drone"], 2, 0, 2)
print("query_feature")
gallery_feature = extract_feature(model, dataloaders["satellite"], 2, 0, 1)


query_img_list = image_datasets["drone"].imgs
gallery_img_list = image_datasets["satellite"].imgs

result = {}
for i in range(len(query_img_list)):
    
    query = query_feature[i].view(-1, 1)
    score = torch.mm(gallery_feature, query)
    score = score.squeeze(1).cpu()
    index = np.argsort(score.numpy())
    index = index[::-1].tolist()
    max_score_list = index[0:10]
    query_img = query_img_list[i][0]
    most_correlative_img = []
    for index in max_score_list:
        most_correlative_img.append(gallery_img_list[index][0])
    result[query_img] = most_correlative_img
    

import pandas as pd
matching_table = pd.DataFrame(result)
print(matching_table)
matching_table.to_csv("result.csv")