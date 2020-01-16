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


First Update Date: Nov. 12, 2019
