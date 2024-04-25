from __future__ import print_function, division

import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from pytorch_metric_learning import losses, miners
import model_
from utils import get_yaml_value, parameter, create_dir, save_feature_network, setup_seed
from Preprocessing import create_U1652_dataloader
import random
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
# torch.cuda.manual_seed(random.randint(1, 100))
setup_seed()
cudnn.benchmark = True



def one_LPN_output(outputs, labels, criterion, block):
    # part = {}

    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    # print(len(outputs))
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)

    return preds, loss


def train():
    classes = get_yaml_value("classes")
    num_epochs = get_yaml_value("num_epochs")
    drop_rate = get_yaml_value("drop_rate")
    lr = get_yaml_value("lr")
    LPN = get_yaml_value("LPN")
    weight_decay = get_yaml_value("weight_decay")
    model_name = get_yaml_value("model")
    batchsize = get_yaml_value("batch_size")
    weight_save_path = get_yaml_value("weight_save_path")
    block = get_yaml_value("block")
    fp16 = get_yaml_value("fp16")

    all_block = block

    dataloaders, image_datasets = create_U1652_dataloader()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}

    print(dataset_sizes)
    class_names = image_datasets['satellite'].classes
    print(len(class_names))

    model = model_.EVA(classes, drop_rate, block).cuda()
    # model.load_state_dict(torch.load("/root/autodl-tmp/weights/Modern_1652_2023-06-23-03:45:26/net_059.pth"))

    if LPN:
        ignored_params = list()
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * lr}]
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': lr})
        optimizer = optim.SGD(optim_params, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        # opt = torchcontrib.optim.SWA(optimizer)
    else:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay, momentum=0.9, nesterov=True)

    if fp16:
        # from apex.fp16_utils import *
        from apex import amp, optimizers
        model, optimizer_ft = amp.initialize(model, optimizer, opt_level="O2")

    criterion = nn.CrossEntropyLoss()
    # criterion1 = nn.KLDivLoss()
    # circle = circle_loss.CircleLoss(m=0.4, gamma=80)
    criterion_func = losses.TripletMarginLoss(margin=0.3)
    infoNCE_loss =losses.NTXentLoss(temperature=0.1) 
    
    # miner = miners.MultiSimilarityMiner()
    miner = miners.TripletMarginMiner()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # swa_scheduler = SWALR(optimizer, swa_lr=0.01)
    # swa_start = 5
    print("Dataloader Preprocessing Finished...")
    MAX_LOSS = 10
    print("Training Start >>>>>>>>")
    weight_save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    dir_model_name = model_name + "_" + str(1652) + "_" + weight_save_name
    save_path = os.path.join(weight_save_path, dir_model_name)
    create_dir(save_path)
    print(save_path)
    parameter("name", dir_model_name)
    warm_epoch = 5
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / batchsize) * warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        since = time.time()

        running_loss = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        total1 = 0.0
        total2 = 0.0
        model.train(True)
        for data1, data2 in zip(dataloaders["satellite"], dataloaders["drone"]):
            input1, label1 = data1
            input2, label2 = data2

            input1 = input1.to(device)
            input2 = input2.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            total1 += label1.size(0)
            total2 += label2.size(0)

            optimizer.zero_grad()
            # output1, output2, feature1, feature2, lpn_1, lpn_2 = model(input1, input2)
            output1, output2, feature1, feature2 = model(input1, input2)

            # fnorm = torch.norm(feature1, p=2, dim=1, keepdim=True) * np.sqrt(all_block)
            # fnorm2 = torch.norm(feature2, p=2, dim=1, keepdim=True) * np.sqrt(all_block)
            # feature1 = feature1.div(fnorm.expand_as(feature1))
            # feature2 = feature2.div(fnorm2.expand_as(feature2))

            # before lpn
            # Identity loss
            # lpn_preds1, lpn_loss1 = one_LPN_output(lpn_1, label1, criterion, all_block)
            # lpn_preds2, lpn_loss2 = one_LPN_output(lpn_2, label2, criterion, all_block)

            # after lpn
            # Identity loss
            # print(output1[0].shape, label1.shape)
            loss1 = loss2 = loss3 = loss4 = loss6 = loss5 = 0
            if LPN:
                # print(len(output1))
                preds1, loss1 = one_LPN_output(output1[2:], label1, criterion, all_block)
                preds2, loss2 = one_LPN_output(output2[2:], label2, criterion, all_block)

                loss3 = criterion(output1[0], label1)
                loss4 = criterion(output2[0], label2)

                loss5 = criterion(output1[1], label1)
                loss6 = criterion(output2[1], label2)
                # _, preds1 = torch.max(output1[1].data, 1)
                # _, preds2 = torch.max(output2[1].data, 1)
                # print(loss)
            else:
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)

                _, preds1 = torch.max(output1.data, 1)
                _, preds2 = torch.max(output2.data, 1)


            # Identity loss
            loss = loss1 + loss2
            # loss += lpn_loss1 + lpn_loss2
            # loss = 0
            # circle loss
            # preds1, loss1 = one_LPN_output(output1, label1, circle, block + 3)
            # preds2, loss2 = one_LPN_output(output2, label2, circle, block + 3)

            # loss += circle(*circle_loss.convert_label_to_similarity(feature1, label1)) / now_batch_size
            # loss += circle(*circle_loss.convert_label_to_similarity(feature2, label2)) / now_batch_size
            # loss = loss1 + loss2


            
            # Triplet loss
            hard_pairs = miner(feature1, label1)
            hard_pairs2 = miner(feature2, label2)
            # loss += criterion_func(feature1, label1, hard_pairs) + \
            #         criterion_func(feature2, label2, hard_pairs2)
            
            # infoNCE loss
            loss += infoNCE_loss(feature1, label1, hard_pairs) + \
                    infoNCE_loss(feature2, label2, hard_pairs2)

            if epoch < warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up
            if fp16:  # we use optimizer to backward loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects1 += preds1.eq(label1.data).sum()
            running_corrects2 += preds2.eq(label2.data).sum()
            # print(loss.item(), preds1.eq(label1.data).sum(), preds2.eq(label2.data).sum())

        scheduler.step()
        epoch_loss = running_loss / len(class_names)
        satellite_acc = running_corrects1 / total1
        drone_acc = running_corrects2 / total2
        time_elapsed = time.time() - since

        print('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s' \
              .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100, time_elapsed))

        if drone_acc > 0.95 and satellite_acc > 0.95:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 10):
                MAX_LOSS = epoch_loss
                save_feature_network(model, dir_model_name, epoch + 1)
                print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))


if __name__ == '__main__':
    from U1652_test_and_evaluate import eval_and_test

    train()
    eval_and_test(448)
