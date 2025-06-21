# AI Spark

AI Spark is a project developed for the 2024 AI Factory wildfire detection satellite image segmentation competition.

## Objective  
Develop a deep learning model that accurately segments wildfire areas from satellite images using AI.

## Approach  
- Due to the nature of satellite images (small objects, mostly background), we experimented with Transformer-based models (e.g., TransUNet), which are considered more suitable than CNN-based models.
- Various segmentation models including TransUNet were applied, and experiments were conducted with data augmentation and different loss functions.

## Project Structure

```
datasets/         # Dataset classes (Spark, baseline, etc.)
loss/             # Custom loss functions and functional implementations
models/
  └─ TransUNet/   # TransUNet and related code (Apache 2.0 License)
trainer/          # Trainer class for training and evaluation
transforms/       # Data augmentation and transformation functions
train.py          # Main training script
test.py           # Test/debugging script
logger.py         # (Empty, for custom logging if needed)
```

## Main Files

- [`train.py`](train.py): Runs the full training pipeline. Handles dataset loading, model creation, and training/validation loops.
- [`test.py`](test.py): Script for experiments and debugging.
- [`datasets/Spark.py`](datasets/Spark.py): PyTorch Dataset class for loading and augmenting data.
- [`loss/loss.py`](loss/loss.py), [`loss/functional.py`](loss/functional.py): Custom loss functions such as BCE, IoU, etc.
- [`trainer/Trainer.py`](trainer/Trainer.py): Trainer class for training, validation, testing, and Tensorboard logging.
- [`models/TransUNet`](models/TransUNet/): TransUNet and related code (Apache 2.0 License).
- [`transforms/transforms.py`](transforms/transforms.py): Collection of data augmentation and preprocessing functions.  
  - Provides various augmentations (horizontal/vertical flip, rotation, etc.) and PyTorch tensor conversion based on Albumentations for both images and masks.
  - Used to increase data diversity and improve model generalization.

## Data Preparation

1. Download the multi-organ dataset from the [official Synapse site](https://www.synapse.org/#!Synapse:syn3193805/wiki/), then convert to numpy format and normalize.
2. Alternatively, use the [BTCV preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) provided by the official TransUNet repository.
3. Example data folder structure:
    ```
    data/
      train_img/
      train_mask/
      train_meta.csv
      test_meta.csv
    ```
    Or for TransUNet format:
    ```
    data/
      Synapse/
        train_npz/
        test_vol_h5/
    ```

## Environment Setup

- Python 3.7 or higher recommended
- Install required packages:
  ```
  pip install -r models/TransUNet/requirements.txt
  ```

## How to Run

### Training
```bash
python train.py
```
- Trains with the TransUNet model by default. To use other models, you need to implement and connect them manually.

### Testing/Experiment
```bash
python test.py
```
- Allows quick experiments and visualization on a subset of the dataset.

## Reference/Citation

- TransUNet Paper: [arXiv:2102.04306](https://arxiv.org/pdf/2102.04306.pdf)
- TransUNet Official Repository: [https://github.com/Beckschen/TransUNet](https://github.com/Beckschen/TransUNet)
- This project follows the [Apache 2.0 License](models/TransUNet/LICENSE) of TransUNet.

## Citation

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
