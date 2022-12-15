# StorSeismic: An approach to pre-train a neural network to store seismic data features
This repository contains codes and resources to reproduce experiments of StorSeismic in Harsuko and Alkhalifah, 2020.

## Requirements
We use [RAdam](https://github.com/LiyuanLucasLiu/RAdam) as the default optimizer. To install this, use:
```
pip install git+https://github.com/LiyuanLucasLiu/RAdam
```

## Instruction

| No | Notebook name |Description |
| --- | --- | --- |
| 1 | [nb0_1_data_prep_pretrain.ipynb](https://github.com/swag-kaust/storseismic/blob/main/nb0_1_data_prep_pretrain.ipynb) | Create pre-training data |
| 2 | [nb0_2_data_prep_finetune.ipynb](https://github.com/swag-kaust/storseismic/blob/main/nb0_2_data_prep_finetune.ipynb) | Create fine-tuning data |
| 3 | [nb1_pretraining.ipynb](https://github.com/swag-kaust/storseismic/blob/main/nb1_pretraining.ipynb) | Pre-training of StorSeismic |
| 4 | [nb2_1_finetuning_denoising.ipynb](https://github.com/swag-kaust/storseismic/blob/main/nb2_1_finetuning_denoising.ipynb) | Example of fine-tuning task: denoising |
| 5 | [nb2_2_finetuning_velpred.ipynb](https://github.com/swag-kaust/storseismic/blob/main/nb2_2_finetuning_velpred.ipynb) | Example of fine-tuning task: velocity estimation |

## References
Harsuko, R., & Alkhalifah, T. A. (2022). StorSeismic: A new paradigm in deep learning for seismic processing. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-15.

## Citation
Citations are very welcomed. This work can be cited using:
```
@article{harsuko2022storseismic,
  title={StorSeismic: A new paradigm in deep learning for seismic processing},
  author={Harsuko, Randy and Alkhalifah, Tariq A},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--15},
  year={2022},
  publisher={IEEE}
}
```
