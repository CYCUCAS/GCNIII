# Wide & Deep Learning for Node Classification
This repository contains a PyTorch implementation of "Wide & Deep Learning for Node Classification".

## Dependencies
- CUDA 12.1
- python 3.12.7
- numpy 1.26.4
- pytorch 2.2.1
- dgl 2.3.0

## Experiments
- To replicate the semi-supervised results, please run the following script:
```sh
sh semi_train.sh
```
- To replicate the full-supervised results, please run the following script:
```sh
sh full_train.sh
```
- To replicate the inductive results, please run the following script:
```sh
sh inductive.sh
```
- To replicate the ablation results, please run the following script:
```sh
sh ablation.sh
```

## Reference

If you find our codes useful, please consider citing our work. Thanks!

```
@article{chen2025widedeeplearning,
      title={Wide & Deep Learning for Node Classification}, 
      author={Yancheng Chen and Wenguo Yang and Zhipeng Jiang},
      journal={arXiv preprint arXiv:2505.02020},
      year={2025}
}
```
