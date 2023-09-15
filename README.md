# RealArch

This repository implements estimation models and schedule algorithm in the paper RealArch: A Real-Time Scheduler for Mapping Multi-Tenant DNNs on Multi-Core Accelerators (ICCD'23)

## Getting Started

### Requirements

-  For software experiments
   -  Python >= 3.7
   -  PyTorch >= 1.8.0
   -  Torchvision >= 1.8.0
### Installation

1.  Clone or download this repository
2.  Create a virtual environment (either [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.anaconda.com/anaconda/install/index.html)) with a Python version of at least 3.7.
3.  Install dependent Python packages: `pip install -r requirements.txt`

## Experiment Workflow

1.  Create estimation result for each network 

    We provide `alexnet.py`, `vgg.py`, and `resnet.py` to verify DM and EX of corresponding three types of networks. For example, to get estimation result on alenxet, you can execute `python alexnet.py`. The results will be stored into `estimation_result_csv/alexnet_estimation.csv`.


2.  Running scheduling algorithm.

    You can get the performance of Round-Robin, AI-MT, and RealArch scheduling algorithm by executing `python run_scheduler.py`. The results will be stored into `exp_res/output_cycle.csv`. Note that you can adapt configurations in `test()` function to get more experiment results. For example, you can specify the combination of input networks by `net_select`.

3. Comparasion of MAGMA and RealArch.

    We provide search space comapasion between MAGMA and RealArch. Please execute `python eval_space.py`. The results will be stored into `exp_res/space_result.csv`

## Citation

Xuhang Wang, Zhuoran Song and Xiaoyao Liang. RealArch: A Real-Time Scheduler for Mapping Multi-Tenant DNNs on Multi-Core Accelerators (ICCD'23)







