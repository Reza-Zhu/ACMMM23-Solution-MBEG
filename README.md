# ACMMM23-Solution-MBEG

## Modern Backbone for Efficient Geo-localization

### Code Overview:

- config: settings.yaml
- model definitions: model_.py
- train: train.py 
- distillation knowledge: distill_train.py
- test: U1652_test_and_evaluate.py
- prepare dataset: Preprocessing.py
- multiply queries: multi.py
- draw heat map: draw_cam_ViT.py

- predict University160k : predict.py / predict.ipynb
- export answer.txt : export.ipynb

## Model weights
Baidu Cloud Disk Link: https://pan.baidu.com/s/1k1z90EyLaL85PqeSxxlWOw?pwd=1652 提取码: 1652 

- MBEG-L1 for University-1652: MBEG-L1-1652.pth
   - Drone -> Satellite: Recall@1:  92.50   AP: 93.75
   - Satellite -> Drone: Recall@1:  94.15   AP: 91.57

- MBEG-L1 for University-160k: MBEG-L1-160k.pth
   - Drone -> Satellite: Recall@1:  98.94	Recall@5:99.80	 Recall@10:99.84

- MBEG-L2 for University-1652: MBEG-L2-1652.pth
   - Drone -> Satellite: Recall@1:  89.78   AP: 91.53
   - Satellite -> Drone: Recall@1:  92.01   AP: 88.81
   
- MBEG-B1 for University-1652: MBEG-B1-1652.pth
   - Drone -> Satellite: Recall@1:  87.48   AP: 89.36
   - Satellite -> Drone: Recall@1:  90.73   AP: 86.29

- MBEG-B2 for University-1652: MBEG-B2-1652.pth
   - Drone -> Satellite: Recall@1:  88.16   AP: 89.97
   - Satellite -> Drone: Recall@1:  93.01   AP: 87.64

- MoblieViT-KD for University-1652: MobileViT-Student-1652.pth
   - Drone -> Satellite: Recall@1:  80.57   AP: 83.44
   - Satellite -> Drone: Recall@1:  88.44   AP: 80.45

## Related Works

### University-1652 Benchmark
https://github.com/layumi/University1652-Baseline

### ACM MM23 Workshop: UAVs in Multimedia: Capturing the World from a New Perspective
https://www.zdzheng.xyz/ACMMM2023Workshop

### EVA Series backbone 
https://github.com/baaivision/EVA
[https://huggingface.co/Yuxin-CV/EVA-02](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)

### MobileViT
https://github.com/apple/ml-cvnets
[https://huggingface.co/timm/mobilevitv2_200.cvnets_in22k_ft_in1k_384]
(https://huggingface.co/timm/mobilevitv2_200.cvnets_in22k_ft_in1k_384)


