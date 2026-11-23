# OCT-Based-Retinal-Disease-Classification-Using-Deep-Learning

An AI-powered retinal disease classification project that uses Optical Coherence Tomography (OCT) images and deep learning techniques to classify retinal conditions into four categories: **CNV, DME, DRUSEN, and NORMAL**.

The project explores neural networks, Convolutional Neural Networks (CNNs), transfer learning, model fine-tuning, performance evaluation, and Grad-CAM explainability. A Streamlit web application provides an interface for uploading OCT images and viewing model predictions.

> **Disclaimer:** This project is developed for educational and research purposes. It is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 1. Project Overview

Retinal diseases can affect vision and may require early medical attention. Optical Coherence Tomography (OCT) is a medical imaging technique that produces cross-sectional images of the retina.

This project applies deep learning techniques to OCT images to identify four retinal image categories.

The main objective is to build and evaluate deep learning models for automated retinal image classification and provide an interactive application for testing predictions.

### Classification Categories

| Class | Description |
|---|---|
| CNV | Choroidal Neovascularization |
| DME | Diabetic Macular Edema |
| DRUSEN | Drusen |
| NORMAL | Normal retinal OCT image |

---

## 2. Project Objectives

- Understand the fundamentals of neural networks through NumPy implementation.
- Develop a custom Convolutional Neural Network from scratch.
- Implement transfer learning using pretrained deep learning architectures.
- Compare different model architectures.
- Apply fine-tuning to improve model performance.
- Evaluate models using classification metrics.
- Visualize model attention using Grad-CAM.
- Develop an interactive Streamlit application for image classification.

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Neural network implementation |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Model evaluation and metrics |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and pretrained models |
| Pillow | Image processing |
| Hugging Face Datasets | Dataset loading |
| Streamlit | Web application |
| Grad-CAM | Model explainability |

---

## 4. Dataset

**Dataset:** Kermany2017 OCT Dataset

**Source:** [Hugging Face – Kermany2017 OCT Dataset](https://huggingface.co/datasets/zacharielegault/Kermany2017-OCT)

The dataset is based on the retinal OCT dataset introduced by Kermany et al. (Cell, 2018).

### Dataset Information

| Dataset Split | Number of Images |
|---|---:|
| Training | 108,309 |
| Testing | 1,000 |
| Total | 109,309 |

The dataset contains grayscale OCT images belonging to four classes:

- CNV
- DME
- DRUSEN
- NORMAL

The notebook loads the dataset automatically using the Hugging Face Datasets library.

**Note:** The notebook uses a balanced subset of 1,000 images per class by default for training experiments. The full training dataset can be used by setting `MAX_TRAIN_PER_CLASS = None` in the notebook configuration.

The official test split is retained for model evaluation.

---

## 5. Project Workflow

The project follows these major stages:

1. Dataset loading and exploration.
2. Image preprocessing.
3. Data visualization and class distribution analysis.
4. NumPy-based neural network implementation.
5. Custom CNN development.
6. Transfer learning using pretrained architectures.
7. Model fine-tuning and regularization.
8. Model evaluation and comparison.
9. Grad-CAM visualization.
10. Best model saving.
11. Streamlit application development.

---

## 6. Image Preprocessing

The OCT images undergo preprocessing before being passed to deep learning models.

The preprocessing operations include:

- Resizing images to 224 × 224 pixels.
- Converting grayscale images into RGB format.
- Image augmentation.
- Image normalization.
- Creating PyTorch datasets and DataLoaders.

These operations prepare the images for training and evaluation.

---

## 7. Deep Learning Models Implemented

### 7.1 NumPy Neural Network

A basic neural network is implemented using NumPy to understand the mathematical foundations of deep learning.

Concepts covered:

- Forward propagation.
- Activation functions.
- Loss calculation.
- Backpropagation.
- Weight updates.
- ReLU, Sigmoid and Tanh activation functions.

### 7.2 Custom CNN

A Convolutional Neural Network is developed from scratch for retinal OCT image classification.

The custom CNN demonstrates how convolutional layers and other neural network components can learn visual patterns from medical images.

### 7.3 Transfer Learning

Four pretrained deep learning architectures are implemented using PyTorch and Torchvision.

| Model | Architecture |
|---|---|
| ResNet18 | Residual Neural Network |
| VGG16 | Visual Geometry Group Network |
| MobileNetV2 | Lightweight Convolutional Network |
| DenseNet121 | Densely Connected Convolutional Network |

Transfer learning is used to leverage pretrained image features for OCT image classification.

### 7.4 Fine-Tuning

Fine-tuning is applied to pretrained models by adjusting selected model parameters using the OCT training data.

This allows the pretrained architectures to adapt their learned features to the retinal disease classification task.

---

## 8. Regularization and Training Techniques

The project explores several techniques to improve training stability and reduce overfitting:

- Dropout
- L2 Weight Decay
- Batch Normalization
- Early Stopping
- Hyperparameter Experiments

The notebook also explores batch size and learning rate configurations.

---

## 9. Model Evaluation

The trained models are evaluated using the unseen test dataset.

The following evaluation metrics are used:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures the ability to identify actual class samples |
| F1-Score | Balances precision and recall |
| Confusion Matrix | Shows correct and incorrect class predictions |
| Inference Time | Measures prediction speed |

The notebook compares the implemented architectures using evaluation results.

The best model is selected based on validation accuracy and saved for later use.

*Actual performance values should be taken from the completed notebook evaluation rather than assumed.*

---

## 10. Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize image regions that influence a model's classification prediction.

Grad-CAM helps understand which regions of an OCT image contribute to the model's output.

It generates heatmap visualizations that can be compared with the original OCT images.

---

## 11. Streamlit Web Application

A Streamlit application is developed to provide an interactive interface for retinal OCT image classification.

### Application Features

- Upload OCT images.
- Preview the uploaded image.
- Predict the retinal image category.
- Display prediction confidence.
- Generate Grad-CAM visualizations.

The application uses the trained `best_model.pt` file for predictions.

---

## 12. Project Structure

```text
OCT-Based-Retinal-Disease-Classification/
│
├── OCT_Retinal_Disease_Detection.ipynb
│
├── streamlit_app.py
│
├── requirements.txt
│
├── README.md
│
└── best_model.pt
```

**File descriptions:**

| File | Description |
|---|---|
| `OCT_Retinal_Disease_Detection.ipynb` | Main notebook containing data processing, model development, training, evaluation and Grad-CAM |
| `streamlit_app.py` | Streamlit application for OCT image classification |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `best_model.pt` | Saved trained model weights required by the Streamlit application |

**Important:** `best_model.pt` is generated by executing the notebook. It is not included in the original GitHub repository.

---

## 13. Installation and Setup

### Prerequisites

- Python 3.10 or compatible Python environment.
- pip package manager.
- Google Colab or a local Python development environment.
- GPU recommended for model training.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/OCT-Based-Retinal-Disease-Classification.git
```

### Step 2: Navigate to the Project Folder

```bash
cd OCT-Based-Retinal-Disease-Classification
```

### Step 3: Create a Virtual Environment (Recommended)

Windows:

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 5: Train the Model

Open `OCT_Retinal_Disease_Detection.ipynb` using Google Colab or Jupyter Notebook.

For Google Colab:

1. Upload the notebook.
2. Select a GPU runtime if available.
3. Execute the notebook cells in order.
4. Complete the model training and evaluation.
5. Generate and download `best_model.pt`.

### Step 6: Run the Streamlit Application

Ensure that `best_model.pt` is placed in the same directory as `streamlit_app.py`.

Run:

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Upload an OCT image to view its predicted category and visualization.

---

## 14. Model File and Dataset Notes

- The dataset is downloaded automatically by the notebook.
- The complete image dataset is not stored in this GitHub repository.
- The trained model file is generated after notebook execution.
- The Streamlit application requires the trained `best_model.pt` file.
- Large model files should be hosted separately using GitHub Releases, Git LFS or another suitable storage service.

---

## 15. Limitations

- The project is an educational deep learning implementation.
- Training results depend on the selected dataset subset, model configuration and computing environment.
- The dataset does not provide patient identifiers, so images from the same patient may appear across training and validation subsets.
- Performance on OCT scans from other hospitals, imaging devices and patient populations has not been clinically validated.
- Grad-CAM visualizations indicate model attention and do not establish medical reasoning or diagnosis.

---

## 16. Future Enhancements

- Improve model generalization using larger and more diverse datasets.
- Experiment with additional deep learning architectures.
- Optimize inference speed.
- Improve the Streamlit user interface.
- Deploy the application on a cloud platform.
- Evaluate models using additional independent datasets.
- Explore further explainable AI techniques.

---

## 17. References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.e9.  
   https://doi.org/10.1016/j.cell.2018.02.010

2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html

3. Torchvision Documentation: https://pytorch.org/vision/stable/index.html

4. Streamlit Documentation: https://docs.streamlit.io/

5. Hugging Face Datasets: https://huggingface.co/docs/datasets/

---

## 18. Author

**Project:** OCT-Based Retinal Disease Classification Using Deep Learning

**Domain:** Artificial Intelligence and Machine Learning

**Technologies:** Python, PyTorch, CNN, Transfer Learning, Grad-CAM, Streamlit

---

⭐ If you find this project useful, consider giving the repository a star!
