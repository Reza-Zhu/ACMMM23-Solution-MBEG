import torch
import os
from utils import get_yaml_value, parameter, create_dir, save_feature_network
from torchvision import datasets, transforms

classes = get_yaml_value("classes")
batchsize = get_yaml_value("batch_size")

data_dir = get_yaml_value("dataset_path")
image_size = get_yaml_value("image_size")


def create_U1652_dataloader():
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'satellite': transforms.Compose(transform_satellite_list)}
    image_datasets = {}
    image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'train', 'satellite'),
                                                       data_transforms['satellite'])
    image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'train', 'drone'),
                                                   data_transforms['train'])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=4, pin_memory=True)
                   # 8 workers may work faster
                   for x in ['satellite', 'drone']}
    return dataloaders, image_datasets


if __name__ == "__main__":
    # Cross_Dataset("../Datasets/SUES-200/Training/150", 224)
    dataloaders, image_datasets = create_U1652_dataloader()
    print(image_datasets['satellite'].classes)
    # for img, label in dataloaders['satellite']:
    #     print(img, label)
    #     break
    U1652_path = "/media/data1/University-Release/University-Release/train"
    # Cross_Dataset_1652(U1652_path, 224)

