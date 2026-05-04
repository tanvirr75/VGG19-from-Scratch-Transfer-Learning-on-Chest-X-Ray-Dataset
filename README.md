# VGG19-from-Scratch-Transfer-Learning-on-Chest-X-Ray-Dataset
Implementation of VGG-19 architecture from scratch and transfer learning using VGG16, ResNet50, and MobileNetV2 for pneumonia classification on the Chest X-Ray dataset using TensorFlow/Keras.

# VGG19 from Scratch & Transfer Learning

Classification of chest X-ray images using a custom VGG-19 implementation and transfer learning with pretrained Keras models.

---

## Dataset

**Chest X-Ray Images (Pneumonia)** by Paul Mooney — [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- 5,863 chest X-ray images
- 2 classes: NORMAL and PNEUMONIA
- Pre-split into train / val / test sets

---



### Part 1: VGG-19 from Scratch
- Built the full VGG-19 architecture using TensorFlow/Keras Sequential API
- 16 convolutional layers + 3 fully connected layers = 19 layers total
- Trained on the chest X-ray dataset for 10 epochs
- Achieved ~95% training accuracy
- Plotted accuracy and loss curves, saved the model as `.h5`

### Transfer Learning with Pretrained Models
Applied and compared 3 pretrained models from Keras Applications on the same dataset:

| Model | Accuracy | F1 Score |
|---|---|---|
| VGG16 | 88.62% | 0.88 |
| MobileNetV2 | 86.22% | 0.85 |
| ResNet50 | 73.08% | 0.70 |

**Best performing model: VGG16**

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
- Kaggle Notebooks (GPU: Tesla P100)

---

## How to Run

1. Open the `.ipynb` file in Kaggle or Google Colab
2. Add the dataset from [this Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. Run all cells in order

---

## Files

```
├── lab4_notebook.ipynb       # Main notebook with all code and outputs
├── vgg19_scratch.h5          # Saved VGG-19 model
├── vgg19_plots.png           # Accuracy and loss plots
├── model_comparison.png      # Bar chart comparing all 3 models
└── README.md
```
