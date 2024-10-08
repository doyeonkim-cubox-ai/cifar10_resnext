# cifar10_resnext
간단한 이미지 분류 모델입니다.

### Introduction

[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431) 논문을 읽고 resnext model을 직접 구현해보았습니다.
학습 데이터셋은 cifar10을 사용했습니다.


### Requirements
After cloning the repo, run this line below:
```
pip install -r requirements.txt
```

### Usage

##### 1. train & test model
```
mkdir model
python -m cifar10_resnext.train -model {모델}
python -m cifar10_resnext.test -model {모델}
# inference
python -m cifar10_resnext.inference -model {모델} -img {이미지 경로}
```

##### 2. statistics
| model              | # layers | # params | error(%) |
|--------------------|:---------|----------|----------|
| Wide ResNet(28-10) | 29       | 36.5M    | 4.19     |
| ResNeXt29(8x64d)   | 29       | 34.4M    | 4.30     |
| ResNeXt29(16x64d)  | 29       | 68.1M    | 4.05     |


##### 3. plots
- testing error(resnext29(8x64d) & resnext29(16x64d))

<img src="./result/resnext.png" width="400" height="300">

- testing error(wresnet)

<img src="./result/wresnet_test.png" width="400" height="300">

- testing error(resnext29(8x64d))

<img src="./result/resnext29(8x64d)_test.png" width="400" height="300">

- testing error(resnext29(16x64d))

<img src="./result/resnext29(16x64d)_test.png" width="400" height="300">
