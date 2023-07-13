from __future__ import print_function, division

import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from pytorch_metric_learning import losses, miners
import torch.nn.functional as F

from sklearn.decomposition import PCA
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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

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


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    


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
    

    Teacher = model_.EVA(classes, drop_rate).cuda()
    Teacher.load_state_dict(torch.load("/root/autodl-tmp/weights/Modern_1652_2023-06-23-03:45:26/net_059.pth"))
    
    Student = model_.MobileViT(classes, drop_rate).cuda()
    TS_criterion = DistillKL(4)
    Cosine_criterion = nn.CosineSimilarity(dim=1)
    if LPN:
        ignored_params = list()
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(Student, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * lr}]
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(Student, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': lr})
        optimizer = optim.AdamW(optim_params, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        # opt = torchcontrib.optim.SWA(optimizer)
    else:
        ignored_params = list(map(id, Student.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, Student.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': Student.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay, momentum=0.9, nesterov=True)

    if fp16:
        # from apex.fp16_utils import *
        from apex import amp, optimizers
        Student, optimizer_ft = amp.initialize(Student, optimizer, opt_level="O2")
        
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()
    # criterion1 = nn.KLDivLoss()
    infoNCE_loss =losses.NTXentLoss(temperature=0.1) 
    # circle = circle_loss.CircleLoss(m=0.4, gamma=80)
    criterion_func = losses.TripletMarginLoss(margin=0.3)
    miner = miners.MultiSimilarityMiner()

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
        Student.train(True)
        for data1, data2 in zip(dataloaders["satellite"], dataloaders["drone"]):
            input1, label1 = data1
            input2, label2 = data2
            resized_input1 = F.interpolate(input1, size=(448, 448), mode='bilinear', align_corners=False)
            resized_input2 = F.interpolate(input2, size=(448, 448), mode='bilinear', align_corners=False)
            
            resized_input1 =resized_input1.to(device)
            resized_input2 =resized_input2.to(device)
            
            input1 = input1.to(device)
            input2 = input2.to(device)
            
            label1 = label1.to(device)
            label2 = label2.to(device)
            total1 += label1.size(0)
            total2 += label2.size(0)

            optimizer.zero_grad()
            
            with torch.no_grad():
                output11, output22, feature11, feature22= Teacher(resized_input1, resized_input2)
                
            # output1, output2, feature1, feature2, lpn_1, lpn_2 = model(input1, input2)
            output1, output2, feature1, feature2 = Student(input1, input2)
            TS_loss = TS_criterion(output1, output11) + TS_criterion(output2, output22)
            # Cosine_loss = (1 - Cosine_criterion(feature1, feature11)) + (1 - Cosine_criterion(feature2, feature22))
            
            loss1 = loss2 = loss3 = loss4 = loss6 = loss5 = Cosine_loss = 0
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
            loss_info = infoNCE_loss(feature1, label1, hard_pairs) + \
                    infoNCE_loss(feature2, label2, hard_pairs2)
            
            loss = (loss1 + loss2 + loss_info) + 0.8 * (TS_loss + Cosine_loss)
            # Triplet loss
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
                save_feature_network(Student, dir_model_name, epoch + 1)
                print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))
                
if __name__ == '__main__':
    from U1652_test_and_evaluate import eval_and_test

    train()
    eval_and_test(384)
