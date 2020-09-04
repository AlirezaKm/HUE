### Hashing-Based Undersampling Ensemble for Imbalanced Pattern Classification Problems

#### Abstract
##### _Undersampling is a popular method to solve imbalanced classification problems. However, sometimes it may remove too many majority samples which may lead to loss of informative samples. In this article, the hashing-based undersampling ensemble (HUE) is proposed to deal with this problem by constructing diversified training subspaces for undersampling. Samples in the majority class are divided into many subspaces by a hashing method. Each subspace corresponds to a training subset which consists of most of the samples from this subspace and a few samples from surrounding subspaces. These training subsets are used to train an ensemble of classification and regression tree classifiers with all minority class samples. The proposed method is tested on 25 UCI datasets against state-of-the-art methods. Experimental results show that the HUE outperforms other methods and yields good results on highly imbalanced datasets._

#### Authors
##### _Wing WY Ng, Shichao Xu, Jianjun Zhang, Xing Tian, Tongwen Rong, Sam Kwong_

#### Publisher
##### _IEEE Transactions on Cybernetics_

#### Publish Date
##### _2020 / 06 / 29_

---

You can access to the paper with the following link: [IEEE](https://ieeexplore.ieee.org/abstract/document/9127877/), [PDF](https://www.researchgate.net/profile/Jianjun_Zhang23/publication/342540019_Hashing-Based_Undersampling_Ensemble_for_Imbalanced_Pattern_Classification_Problems/links/5f00d8caa6fdcc4ca44bf610/Hashing-Based-Undersampling-Ensemble-for-Imbalanced-Pattern-Classification-Problems.pdf)

---

##### Installation
```bash
$ pip install -r ./requirements.txt
```

#### Run
You can read the `run.py` file and do the same as that (e.g. change it how you want) then run the following command:

```bash
$ python ./run.py
```

#### Directories and Files

| Path         | Description     |
| :----------- | :-------------  |
| [data/](data/)   | selected datasets in order to evaluate the implementation |
| [ploting.py](ploting.py)   | Generate samples (circular) and plot and apply ITQ |
| [ensemble.py](ensemble.py)   | Implementation of HashBasedUndersamplingEnsemble |
| [utils.py](utils.py)   | useful functions like prapering data and evaluation |
| [run.py](run.py)   | Using the implemented proposed method on the datasets |


#### Selected Datasets
|  Datasets |
| :------- |
| [abalone](https://archive.ics.uci.edu/ml/datasets/Abalone)  |
| [car](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)  |
| [flare-F](https://sci2s.ugr.es/keel/imbalanced.php)  |
| [glass](https://archive.ics.uci.edu/ml/datasets/glass+identification)  |
| [ILPD](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))  |
| [letter](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)  |
| [seeds](https://archive.ics.uci.edu/ml/datasets/seeds)  |
| [Skin](https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation)  |
| [wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)  |
| [yeast5](https://sci2s.ugr.es/keel/imbalanced.php)  |

------------


##### TODO(s)
- [ ] we need to support labels other than 1 and -1
- [ ] improve the code



