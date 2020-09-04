from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from utils import evaluate
import pandas as pd

DATASETS = dict()

"""Wine Dataset"""
X, y = load_wine(return_X_y=True)
DATASETS.update({
    'Wine': {
        'data': [X, y],
        'extra': {
        }
    }
})

"""Flare-F"""
data = pd.read_csv('data/raw/flare-F.dat', header=None)
objects = data.select_dtypes(include=['object'])
for col in objects.columns:
    if col == len(data.columns) - 1:
        continue
    data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])

DATASETS.update({
    'Flare-F': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {

        }
    }
})

"""Yeast5"""
data = pd.read_csv('data/raw/yeast5.dat', header=None)
DATASETS.update({
    'Yeast5': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Car vGood"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarvGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'vgood'
        }
    }
})


"""Car Good"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'good'
        }
    }
})

"""Seed"""
data = pd.read_csv('data/raw/seeds_dataset.txt', header=None)
DATASETS.update({
    'Seed': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 2
        }
    }
})

"""Glass"""
data = pd.read_csv('data/raw/glass.data', header=None)
DATASETS.update({
    'Glass': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 7
        }
    }
})

"""ILPD"""
data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
data.fillna(data.mean(), inplace=True)

# Encode
data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

DATASETS.update({
    'ILPD': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Yeast5-ERL"""
data = pd.read_csv('data/raw/yeast5.data', header=None)
DATASETS.update({
    'Yeast5-ERL': {
        'data': [data.values[:, 1:-1], data.values[:, -1]],
        'extra': {
            # 'minority_class': 'ME1'
            'minority_class': 'ERL'
        }
    }
})

"""Skin"""
data = pd.read_csv('data/raw/Skin_NonSkin.txt', delimiter='\t', header=None)
DATASETS.update({
    'Skin': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Letter Dataset"""
data = pd.read_csv('data/raw/letter-recognition.data', header=None)
DATASETS.update({
    'Letter': {
        'data': [data.values[:, 1:], data.values[:, 0]],
        'extra': {
            'minority_class': 'A'
        }
    }
})

# """MNIST"""
# data = pd.read_csv('data/raw/mnist_784.csv', header=0)
# DATASETS.update({
#     'DIGITS': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             'minority_class': 4,
#             'n_runs': 1
#         }
#     }
# })

for name, value in DATASETS.items():
    for method in [
        'reciprocal',
        'random',
        'linearity',
        'negexp',
        'limit'
    ]:
        evaluate(
            "{} - Method: {}".format(name, method.title()),
            DecisionTreeClassifier(),
            *value.get('data'),
            **value.get('extra'),
            k=5,
            verbose=True,
            sampling=method
        )
    print("*"*50)
