# 🔬 Blood Smear AI — White Blood Cell Classification

An AI-powered white blood cell classification prototype built using **Python, PyTorch, and Streamlit**.

This project explores how deep learning and computer vision can assist in the classification of white blood cells from peripheral blood smear images.

## 🧬 Project Overview

White blood cell (WBC) examination is an important part of laboratory hematology. Traditional microscopic examination requires trained laboratory professionals and can be time-consuming.

The goal of this project is to develop a simple AI-based prototype that can classify white blood cell images into five categories using a Convolutional Neural Network (CNN).

The project combines a trained deep learning model with a Streamlit web interface, allowing users to upload an image and receive a predicted cell class.

**This is an educational and portfolio project, not a clinically validated diagnostic system.**

## 🎯 Classification Categories

The model is trained to recognize five types of white blood cells:

| Cell type | Description |
|---|---|
| Basophil | Granulocyte involved in inflammatory and allergic responses |
| Eosinophil | Granulocyte associated with allergic responses and parasitic infections |
| Lymphocyte | Includes important cells involved in adaptive and innate immunity |
| Monocyte | White blood cell involved in phagocytosis and immune defense |
| Neutrophil | Granulocyte important in the body's early response to infection |

## 🧠 Model Architecture

A custom Convolutional Neural Network (CNN) was developed using PyTorch.

The architecture includes:

- Convolutional layers for image feature extraction.
- ReLU activation functions.
- Max-pooling layers for spatial dimension reduction.
- Adaptive average pooling.
- Dropout for regularization.
- A fully connected output layer for five-class classification.

**Model configuration:**

- Image size: 128 × 128 pixels
- Batch size: 16
- Training epochs: 10
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Weighted Cross-Entropy Loss

The model uses softmax to produce scores for the five classes.

## 📊 Dataset and Preprocessing

The project uses a peripheral blood cell image dataset organized into five cell categories.

The images were divided into three subsets:

| Dataset split | Images |
|---|---:|
| Training | 7,206 |
| Validation | 1,543 |
| Testing | 1,550 |
| **Total** | **10,299** |

The split was performed separately for each class using a fixed random seed.

Preprocessing includes:
- Resizing images to 128 × 128 pixels.
- Converting images into tensors.
- Normalizing image pixel values.

**Note:** The dataset itself is not included in this GitHub repository. Users must obtain the dataset separately and organize it according to the expected folder structure.

## 📈 Model Evaluation

The evaluation script reported an overall test accuracy of **98.26%**.

The reported class-wise metrics are:

| Cell type | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Basophil | 98.35% | 97.28% | 97.81% |
| Eosinophil | 99.79% | 99.36% | 99.57% |
| Lymphocyte | 98.29% | 93.99% | 96.09% |
| Monocyte | 92.58% | 92.07% | 92.32% |
| Neutrophil | 99.40% | 98.80% | 99.10% |

The confusion matrix is available in `confusion_matrix.png`.

**Evaluation note:** The evaluation output reported 1,549 test images, while the split-folder counts total 1,550. This discrepancy has not yet been resolved, so the reported metrics should be treated as preliminary rather than independently verified results.

## 🖥️ Streamlit Application

The project includes a web interface developed using Streamlit.

Users can:
1. Upload a blood smear image.
2. View the uploaded image.
3. Run the trained classification model.
4. View the predicted cell category.
5. Examine the model score and scores across all five classes.

## ⚙️ Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/anubhavsarraf9-eng/blood-smear-ai.git
cd blood-smear-ai
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision streamlit pillow scikit-learn matplotlib
```

### 4. Run the application

```bash
streamlit run app.py
```

The application should open in your browser.

The trained model file, `best_blood_cell_model.pth`, is included in this repository, so you do not need to train the model again to use the application.

## 📁 Project Structure

```text
blood-smear-ai/
│
├── app.py
├── predict_cell.py
├── train_model.py
├── evaluate_model.py
├── prepare_dataset.py
├── inspect_dataset.py
├── visualize_cells.py
│
├── best_blood_cell_model.pth
├── confusion_matrix.png
├── .gitignore
│
├── data/                  # Dataset (not included)
├── models/
├── notebooks/
└── src/
```

## ⚠️ Limitations

- The model is designed for single-image, five-class classification.
- It does not detect, locate, or count multiple cells in an image.
- Predictions may be affected by image quality, staining, illumination, and differences between microscopes.
- Model scores are not calibrated clinical probabilities.
- The model has not undergone clinical validation or external hospital-based testing.

This application must not be used to make medical diagnoses or treatment decisions.

## 🚀 Future Improvements

- Implement multi-cell detection and classification.
- Explore object detection and image segmentation techniques.
- Improve model generalization across different microscopes and staining conditions.
- Perform external validation using independently collected images.
- Explore explainable AI techniques such as Grad-CAM.
- Investigate potential applications in laboratory research and resource-limited settings.

## 👨‍🔬 Author

**Hariom Sarraf**

BSc Medical Laboratory Technology | Deep Learning | Medical Image Analysis

GitHub: https://github.com/anubhavsarraf9-eng

---

*Built as a personal learning and research-oriented portfolio project exploring the intersection of medical laboratory science, artificial intelligence, and computational biology.*
