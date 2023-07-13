import argparse
import cv2
import numpy as np
import torch
import glob
import re
import os
import hbp_pooling
from einops import rearrange
import matplotlib.pyplot as plt
from utils import get_yaml_value, create_dir

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def draw_heat_map(weights, img_paths, name):

    count = 0
    create_dir(name)
    for weight in weights:
        print(weight)
        model = hbp_pooling.Hybird_ViT(701, 0)

        model.load_state_dict(torch.load(weight))
        model = model.model_1

        model.eval()

        if args.use_cuda:
            model = model.cuda()
        # model.model_1.
        target_layers = [model.blocks[-1].norm1]

        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=args.use_cuda,
                                       reshape_transform=reshape_transform,
                                       ablation_layer=AblationLayerVit())
        else:
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=args.use_cuda,
                                       reshape_transform=reshape_transform)


        for img_path in img_paths:

            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (384, 384))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            # print(input_tensor.shape)

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            targets = [ClassifierOutputTarget(-1)]

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cv2.imwrite(os.path.join(name, name + f'{args.method}_cam_%d_vit.jpg' % count), cam_image)

            # rgb = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
            # plt.figure("black")
            # plt.imshow(rgb)
            # plt.show()
            print(count)
            count += 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam++',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=24, width=24):
    # print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # print(result.shape)
    # result = rearrange(result, "b (h w) y -> b y h w", h=24, w=24)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    # print(result.shape)

    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    paths = ["/home/sues/media/disk2/save_hybrid_weight/HBP_VIT_BLOCK_1652_2022-10-29-19:41:40/net_058.pth"]

    # img_path = "/home/sues/media/disk2/University-Release/University-Release/test/gallery_drone/0010/image-34.jpeg"
    data_path = get_yaml_value("dataset_path")
    # height = 300
    data_path = "/home/sues/media/disk2/Datasets"
    height = 300
    gallery_drone_path = os.path.join(data_path, "Testing", str(height), "gallery_drone")
    gallery_satellite_path = os.path.join(data_path, "Testing", str(height), "gallery_satellite")

    # gallery_drone_path = os.path.join(data_path, "test", "gallery_drone")
    # gallery_satellite_path = os.path.join(data_path, "test", "gallery_satellite")
    gallery_drone_list = glob.glob(os.path.join(gallery_drone_path, "*"))
    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    gallery_satellite_list = glob.glob(os.path.join(gallery_satellite_path, "*"))
    gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    # query_name = 'query_drone'
    # gallery_name = 'gallery_satellite'

    drone_list = []
    satellite_list = []

    for drone_img in gallery_drone_list:
        img_list = glob.glob(os.path.join(drone_img, "*"))
        img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
        for img in img_list:
            drone_list.append(img)

    for satellite_img in gallery_satellite_list:
        img_list = glob.glob(os.path.join(satellite_img, "*"))
        img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
        for img in img_list:
            satellite_list.append(img)

    draw_heat_map(paths, drone_list[:5000], "drone_MBF_SUES")
    draw_heat_map(paths, satellite_list[:], "satellite_MBF_SUES")
