import os
import time
import yaml
from utils import parameter
from train import train
from U1652_test_and_evaluate import eval_and_test


def Auto_tune(drop_rate, learning_rate):
    # for model in model_list:
    #     parameter("model", model)
        for dr in drop_rate:
            parameter("drop_rate", dr)
            for lr in learning_rate:
                parameter("lr", lr)
                # for wd in weight_decay:
                #     parameter("weight_decay", wd)
                with open("settings.yaml", "r", encoding="utf-8") as f:
                    setting_dict = yaml.load(f, Loader=yaml.FullLoader)
                    print(setting_dict)
                    f.close()
                train()
                try:
                    eval_and_test(384)
                except:
                    print("error")
                    continue


learning_rate = [0.001, 0.002, 0.003, 0.005]
drop_rate = [0.3, 0.45]
# weight_decay = [0.0001, 0.0005, 0.001]
Auto_tune(drop_rate, learning_rate)
