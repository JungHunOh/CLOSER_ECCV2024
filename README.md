# CLOSER: Towards Better Representation Learning for Few-Shot Class-Incremental Learning
#### Junghun Oh*, Sunyong Baik*, and Kyoung Mu Lee

Official Pytorch implementation of **"CLOSER: Towards Better Representation Learning for Few-Shot Class-Incremental Learning"** accepted at **ECCV2024**.
[Paper](https://arxiv.org/abs/2410.05627)

If you find this repo useful for your research, please consider citing our paper:
```
@InProceedings{oh2024closer,
  author = {Oh, Junghun and Baik, Sungyong and Lee, Kyoung Mu},
  title = {CLOSER: Towards Better Representation Learning for Few-Shot Class-Incremental Learning},
  booktitle = {IEEE/CVF European conference on computer vision (ECCV)},
  year = {2024}
}
```

## Abstract
Aiming to incrementally learn new classes with only few samples while preserving the knowledge of base (old) classes, few-shot class-incremental learning (FSCIL) faces several challenges, such as overfitting and catastrophic forgetting. Such a challenging problem is often tackled by fixing a feature extractor trained on base classes to reduce the adverse effects of overfitting and forgetting. Under such formulation, our primary focus is representation learning on base classes to tackle the unique challenge of FSCIL: simultaneously achieving the transferability and discriminability of the learned representation. Building upon the recent efforts for enhancing the transferability, such as promoting the spread of features, we find that trying to secure the spread of features within a more confined feature space enables the learned representation to strike a better balance between the transferability and discriminability. Thus, in stark contrast to priory beliefs that the inter-class distance should be maximized, we claim that the CLOSER different classes are, the better for FSCIL. The empirical results and analysis from the perspective of information bottleneck theory justify our simple yet seemingly counter-intuitive representation learning method, raising research questions and suggesting alternative research directions.

## CLOSER

<img src='./images/Figure1.png' width='2000' height='800'>

For detailed descriptions on the proposed method and experimental results, please refer to the paper.

## Setup (cuda 12.1)
```bash
conda env create -n YourEnv -f dependencies.yaml
conda activate YourEnv
```

## Datasets
Please follow the instruction in [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file#datasets-and-pretrained-models).

## Experiments

### CIFAR100
```bash
python train.py -project closer -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu $gpu --closer --save closer -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 1 --temp 32
```

### miniImageNet
```bash
python train.py -project closer -dataset mini_imagenet -lr_base 0.1 -epochs_base 200 -gpu $gpu --closer --save closer -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 0.5 --temp 32
```

### CUB200
```bash
python train.py -project closer -dataset cub200 -lr_base 0.005 -epochs_base 50 -gpu $gpu --closer --save closer -batch_size_base 256 -seed 1 --ssc_lamb 0.01 --inter_lamb 1.5 --temp 32
```

See the effect of inter-class distance minimization loss by controlling the 'inter_lamb' argument!

## Trained models
We uploaded the trained models using CLOSER [here](https://drive.google.com/drive/folders/10STnlGnLPhxJs4_UMP-nOg54p1SBY_KY?usp=sharing).

For evaluation the models:
```bash
python train.py -project closer -dataset $dataset -gpu $gpu --closer --save closer_trained --eval_only -model_dir $model_dir
```

## Acknowledgment
We reference the following repositories:
- [fscil](https://github.com/xyutao/fscil)
- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [corinfomax-ssl](https://github.com/serdarozsoy/corinfomax-ssl)
- [SupContrast](https://github.com/HobbitLong/SupContrast)

