# Automated Early Glaucoma Identification System (A.E.G.I.S.)

An AI-powered application for early detection of glaucoma from fundus eye images using deep learning.

---

## Overview

This project aims to assist in the early diagnosis of glaucoma by analyzing fundus images and classifying them into:

* **GON+ (Glaucoma Positive)**
* **GON- (Glaucoma Negative)**

The system leverages deep learning techniques to provide accurate predictions along with visual explanations.

---

## Features

* Upload fundus images
* Automatic glaucoma prediction (GON+ / GON-)
* Confidence score for predictions
* Grad-CAM visualization for model interpretability

---

## Tech Stack

* Python
* Streamlit
* PyTorch
* Torchvision
* timm
* OpenCV (headless)
* Albumentations
* NumPy
* Matplotlib
* Grad-CAM
* Pillow

---

## Model

The trained deep learning model is stored as:

swin_glaucoma_model_best.pth

> Due to file size limitations, the model is managed using Git LFS or external hosting.

---

## Dataset

* HYGD (Hillel Yaffe Glaucoma Dataset)

---

## How to Run

### Option 1 — Run via Hugging Face (Recommended)

This project is currently deployed on Hugging Face Spaces.
Simply access the app via the provided link.
Link: https://ghozihumam-swin-transformer.hf.space/
---

### Option 2 — Run Locally

1. Clone the repository:

```bash
git clone https://github.com/ghozihumamridho/glaucoma-detection-swin-Bahtera-Melaju.git
cd glaucoma-detection-swin-Bahtera-Melaju
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Sample Output

**UI**
<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/5e2cbcc8-5e41-485a-af18-44f218ad8c33" />

**Output**
<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/acbbd2a6-161e-4cf5-8890-d6c4df09ccd0" />

---

## Author

**Team BahteraMelaju**

---
