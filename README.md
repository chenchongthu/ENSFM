# ENSFM

This is our implementation of the paper:

*Chong Chen, Min Zhang, Weizhi Ma, Yiqun Liu and Shaoping Ma. 2020. [Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation.](https://chenchongthu.github.io/files/WWW_ENSFM.pdf) 
In TheWebConf'20.*

**Please cite our TheWebConf'20 paper if you use our codes. Thanks!**

```
@inproceedings{chen2020efficient,
  title={Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation},
  author={Chen, Chong and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of The Web Conference},
  year={2020},
}
```

Author: Chong Chen (cstchenc@163.com)

## Baselines

For FM, NFM, ONCF and CFM, we use the implementations released in https://github.com/chenboability/CFM.

For Frappe and Last.fm datasets, the results of FM, DeepFM, NFM, ONCF, and CFM are the same as those reported in [CFM: Convolutional Factorization Machines for Context-Aware Recommendation.](https://www.ijcai.org/proceedings/2019/0545.pdf) since we share exactly the same data splits and experimental settings.


## Environments

- python
- Tensorflow
- numpy
- pandas


## Example to run the codes		

Train and evaluate the model:

```
python ENSFM.py
```
## Suggestions for parameters

Two important parameters need to be tuned for different datasets, which are:
```
parser.add_argument('--dropout', type=float, default=1,
                        help='dropout keep_prob')
parser.add_argument('--negative_weight', type=float, default=0.5,
                        help='weight of non-observed data')
```

Specifically, we suggest to tune "negative_weight" among \[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]. Generally, this parameter is related to the sparsity of dataset. If the dataset is more sparse, then a small value of negative_weight may lead to a better performance.


Generally, the performance of our ENSFM is much better than existing state-of-the-art FM models like NFM, DeepFM, and CFM on Top-K recommendation task. You can also contact us if you can not tune the parameters properly.

First Update Date: May 19, 2020
