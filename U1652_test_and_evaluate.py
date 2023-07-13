# -*- coding: utf-8 -*-
import glob
import os
import time
import timm
import torch
import shutil
import numpy as np
import pandas as pd
import model_
from torch import nn

from utils import fliplr, load_network, which_view, get_id, get_yaml_value, rotate
from torchvision import datasets, models, transforms


if torch.cuda.is_available():
    device = torch.device("cuda:0")


def evaluate(qf, ql, gf, gl):

    query = qf.view(-1, 1)

    score = torch.mm(gf, query)
    # score.shape = (51355,1)
    score = score.squeeze(1).cpu()
    # score.shape = （51355,)
    score = score.numpy()
    # print(score)
    # print(score.shape)

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]


    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    good_index = query_index

    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)

    mask = np.in1d(index, good_index)

    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        # n/sum
        # print("row_good[]", i, rows_good[i])
        # print(precision)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


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


############################### main function #######################################
def eval_and_test(image_size):
    image_size = get_yaml_value("image_size")
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    block = get_yaml_value("block")
    LPN = get_yaml_value("LPN")
    data_dir = get_yaml_value("dataset_path")
    all_block = block
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, 'test', x), data_transforms) for x in
                      ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    
    print(len(image_datasets["query_drone"]))
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  # batch_size=get_yaml_value("batch_size"),
                                                  batch_size=16,
                                                  shuffle=False, num_workers=4) for x in
                   ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    
    # print("Testing Start >>>>>>>>")
    table_path = os.path.join(get_yaml_value("weight_save_path"),
                              get_yaml_value("name") + ".csv")
    save_model_list = glob.glob(os.path.join(get_yaml_value("weight_save_path"),
                                             get_yaml_value('name'), "*.pth"))
    # print(get_yaml_value("name"))
    if os.path.exists(os.path.join(get_yaml_value("weight_save_path"),
                                   get_yaml_value('name'))) and len(save_model_list) >= 1:
        if not os.path.exists(table_path):
            evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])
        else:
            evaluate_csv = pd.read_csv(table_path)
            evaluate_csv.index = evaluate_csv["index"]
        for query in ['drone', "satellite"]:
            for seq in range(-1, 0):
                # net_name = "mae_pretrained"
                model, net_name = load_network(seq=seq)
                net_name = "LPN"
                # model = model_.two_view_net(701, 0.3)
                # model.load_state_dict(torch.load("/home/sues/Reza/SS-Study/LPN/save_model_weight/three_view_lr_0.01_dr_0.5_2/net_119.pth"))
                # print(model)
                # LPN
                if LPN:
                    for i in range(all_block):
                        cls_name = 'classifier' + str(i)
                        c = getattr(model, cls_name)
                        c.classifier = nn.Sequential()
                else:
                    model.classifier.classifier = nn.Sequential()
                # print(net_name)

                model = model.eval()
                model = model.cuda()
                # print(model)
                query_name = ""
                gallery_name = ""

                if query == "satellite":
                    query_name = 'query_satellite'
                    gallery_name = 'gallery_drone'
                elif query == "drone":
                    query_name = 'query_drone'
                    gallery_name = 'gallery_satellite'

                which_query = which_view(query_name)
                which_gallery = which_view(gallery_name)

                print('%s -> %s:' % (query_name, gallery_name))

                # image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path)

                gallery_path = image_datasets[gallery_name].imgs
                query_path = image_datasets[query_name].imgs

                gallery_label, gallery_path = get_id(gallery_path)
                query_label, query_path = get_id(query_path)

                with torch.no_grad():
                    since = time.time()
                    query_feature = extract_feature(model, dataloaders[query_name], all_block, LPN, which_query)
                    gallery_feature = extract_feature(model, dataloaders[gallery_name], all_block, LPN, which_gallery)
                    print(query_feature.shape)
                    print(gallery_feature.shape)

                    time_elapsed = time.time() - since
                    print('Testing complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))

                #     result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
                #               'gallery_path': gallery_path,
                #               'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
                #
                #     scipy.io.savemat('U1652_pytorch_result.mat', result)
                #
                # print(">>>>>>>> Testing END")
                #
                # print("Evaluating Start >>>>>>>>")
                #
                # result = scipy.io.loadmat("U1652_pytorch_result.mat")
                #
                # # initialize query feature data
                # query_feature = torch.FloatTensor(result['query_f'])
                # query_label = result['query_label'][0]
                #
                # # initialize all(gallery) feature data
                # gallery_feature = torch.FloatTensor(result['gallery_f'])
                # gallery_label = result['gallery_label'][0]
                query_feature = query_feature.cuda()
                gallery_feature = gallery_feature.cuda()
                query_label = np.array(query_label)
                gallery_label = np.array(gallery_label)

                # fed tensor to GPU
                query_feature = query_feature.cuda()
                gallery_feature = gallery_feature.cuda()

                # CMC = recall
                CMC = torch.IntTensor(len(gallery_label)).zero_()
                # ap = average precision
                ap = 0.0

                for i in range(len(query_label)):
                    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC += CMC_tmp
                    ap += ap_tmp

                # average CMC

                CMC = CMC.float()
                CMC = CMC / len(query_label)
                # print(len(query_label))
                recall_1 = CMC[0] * 100
                recall_5 = CMC[4] * 100
                recall_10 = CMC[9] * 100
                recall_1p = CMC[round(len(gallery_label) * 0.01)] * 100
                AP = ap / len(query_label) * 100

                evaluate_csv[query_name+"_"+net_name] = [float(recall_1), float(recall_5),
                                                         float(recall_10), float(recall_1p),
                                                         float(AP),
                                                         float(time_elapsed)
                                                         ]
                evaluate_result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f Time::%.2f' % (
                    recall_1, recall_5, recall_10, recall_1p, AP, time_elapsed
                ) + "eva02_small_patch14_336.mim_in22k_ft_in1k"

                # show result and save
                save_path = os.path.join(get_yaml_value("weight_save_path"), get_yaml_value('name'))
                save_txt_path = os.path.join(save_path,
                                             '%s_to_%s_%s_%.2f_%.2f.txt' % (query_name[6:], gallery_name[8:], net_name[:7],
                                                                            recall_1, AP))
                # print(save_txt_path)

                with open(save_txt_path, 'w') as f:
                    f.write(evaluate_result)
                    f.close()

                shutil.copy('settings.yaml', os.path.join(save_path, "settings_saved.yaml"))
                shutil.copy('train.py', os.path.join(save_path, "train.py"))


                # print(round(len(gallery_label)*0.01))
                print(evaluate_result)
        # evaluate_csv["max"] =
        drone_max = []
        satellite_max = []

        for index in evaluate_csv.index:
            drone_max.append(evaluate_csv.loc[index].iloc[:5].max())
            satellite_max.append(evaluate_csv.loc[index].iloc[5:].max())

        evaluate_csv['drone_max'] = drone_max
        evaluate_csv['satellite_max'] = satellite_max
        evaluate_csv.columns.name = "net"
        evaluate_csv.index.name = "index"
        evaluate_csv.to_csv(table_path)
    else:
        print("Don't have enough weights to evaluate!")


if __name__ == '__main__':
    eval_and_test(384)
