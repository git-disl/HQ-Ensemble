# HQ-Ensemble: Hierarchical Ensemble Pruning
-----------------
[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version](https://img.shields.io/badge/version-0.0.1-red.svg?style=flat)]()
<!---
[![Travis Status]()]()
[![Jenkins Status]()]()
[![Coverage Status]()]()
--->

## Introduction

If you find this work useful in your research, please cite the following paper:

**Bibtex**:
```bibtex
@InProceedings{hq-ensemble,
author={Wu, Yanzhao and Liu, Ling},
booktitle={2021 IEEE International Conference on Data Mining (ICDM)}, 
title={{Boosting Deep Ensemble Performance with Hierarchical Pruning}}, 
volume={},
number={},
pages={1433-1438},
month = {Dec.},
year = {2021}
}
```

## Instructions

Following the steps below for using our HQ-Ensemble for efficient ensemble pruning for a dataset \<dataset\>.

1. Install then dependencies in requirements.txt, and obtain the pretrained models for the dataset \<dataset\> according to the model files under \<dataset\> folder.
2. Extract the prediction vectors and labels for \<dataset\> and store them as numpy vectors under \<dataset\>/prediction for testing data and \<dataset\>/train for training data.
3. Set up the environments with env.sh, then execute the HQ-Ensemble.py or naiveDiversityPruning.py file to obtain the corresponding results.

Please refer to our paper and appendix for detailed results.

## Problem


## Installation
    pip install -r requirements.txt

## Supported Platforms


## Development / Contributing


## Issues


## Status


## Contributors

See the [people page](https://github.com/git-disl/HQ-Ensemble/graphs/contributors) for the full listing of contributors.

## License

Copyright (c) 20XX-20XX [Georgia Tech DiSL](https://github.com/git-disl)  
Licensed under the [Apache License](LICENSE).

