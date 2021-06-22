# ReadME

This repository contains implementation of HeadHunter-T proposed in our CVPR 2021 paper, `Tracking Pedestrian Heads in Dense Crowd`. 



## Dataset :

The proposed CroHD dataset can be downloaded from [here](https://motchallenge.net/data/Head_Tracking_21/). Please note that tracking ground truth are available only for training set. Test set annotation are upheld for fair evaluation. Please feel free to submit your method's result on the MOTChallenge server directly. 

## Setup Instructions:

In order to execute this codebase the following requirements need to be satisfied. 

1. Nvidia Driver >= 418
2. Cuda 10.0 is needed if Docker is unavailable.
3. Anaconda.
4. HeadHunter - the head detector to be installed as a python package and the path to weights of pre-trained head detector. 
5. The custom `PyMotMetrics` for evaluation using IDEucl metrics. 
4. Python packages : To install the required python packages;
	```conda env create -f env.yml```


## Instructions to Run: 

1. To run the tracker on a CroHD dataset, 

	```
	python run_mot.py --base_dir /path/to/CroHD/ --cfg_file <your config file> --dataset <test/train> --save_path <directory where results in MOT format can be saved>
	``` 

2. To run the tracker on another dataset where frames are decoded into `.jpg`. Please note that the tracking might fail if 
	attempted at resolution significantly different from what the object detector (HeadHunter) is trained on.

	```
	python run_new.py --base_dir /path/to/frames --save_dir /path/to/save/tracks 
	```

3. To evaluate the MOT tracking accuracies based on existing metrics and the proposed IDEucl metric,

	```
	python evaluation.py --gt_dir /path/to/training/gt --pred_dir /path/to/prediction
	```



## Citation :

In case this code / dataset / work helps in your research, please cite us as,

```
@InProceedings{Sundararaman_2021_CVPR,
    author    = {Sundararaman, Ramana and De Almeida Braga, Cedric and Marchand, Eric and Pettre, Julien},
    title     = {Tracking Pedestrian Heads in Dense Crowd},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3865-3875}
}
```


## Note :
There is a slight discrepancy between the results observed in MOTChallenge website and what is reported in the paper (Table 12, supplementary). This is due to difference in the evaluation scheme between TrackEval and Py-motmetrics. In order to reproduce the results in the paper, please use the py-motmetric based on this [pull request](https://github.com/cheind/py-motmetrics/pull/149)
