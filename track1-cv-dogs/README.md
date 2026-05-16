# Dog Re-Identification using DINOv2

This project implements a prototype dog re-identification (ReID) pipeline using pretrained DINOv2 embeddings and cosine similarity retrieval.

The goal of the system is to determine whether a query image contains the same individual dog as a provided reference image. Unlike breed classification, this task focuses on identifying individual dogs using fine-grained visual features.

The notebook:
- Accepts a reference image and a set of query images
- Extracts image embeddings using DINOv2
- Computes cosine similarity scores
- Returns ranked retrieval results
- Evaluates performance using retrieval and classification metrics
- Visualizes successful matches and failure cases

---

# Repository Structure

```text
track1-cv-dogs/
│
├── dog_reid.ipynb
├── README.md
├── REPORT.md
├── PR_DESCRIPTION.md
│
├── sample_dataset/
│   ├── dog_001/
│   ├── dog_002/
│   └── ...
│
├── outputs/
│   ├── retrieval_examples/
│   ├── failure_cases/
│   └── plots/
│
└── requirements.txt
```

---

# Environment

The project was developed as a Jupyter notebook (`.ipynb`) intended to be run using either:
- Google Colab
- Jupyter Notebook
- VSCode with Jupyter support
- PyCharm Professional
- Any IDE supporting notebook execution

Google Colab is the recommended environment because the notebook automatically installs required dependencies and can easily access datasets stored in Google Drive.

---

# Dataset

The images used for this project were sourced from the DogFaceNet dataset after locating publicly available dataset files online.

A small curated sample dataset is included in the repository under:

```text
track1-cv-dogs/sample_dataset/
```

Each folder represents one individual dog identity:

```text
sample_dataset/
    dog_001/
        img1.jpg
        img2.jpg

    dog_002/
        img1.jpg
        img2.jpg
```

This identity-based structure is required for the re-identification task.

---

# Dataset Setup (Google Colab)

## Step 1 — Upload Dataset to Google Drive

Upload the dataset folder to your Google Drive.

Example structure:

```text
MyDrive/
    dog_reid_dataset/
        dog_001/
        dog_002/
        dog_003/
```

---

## Step 2 — Open the Notebook in Google Colab

Open:

```text
dog_reid.ipynb
```

in Google Colab.

---

## Step 3 — Mount Google Drive

Run the notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize access when prompted.

---

## Step 4 — Update Dataset Path

Set the dataset path variable to your uploaded dataset location:

```python
dataset_dir = "/content/drive/MyDrive/dog_reid_dataset"
```

---

# Installation

The notebook installs required packages automatically using:

```python
!pip install -q torch torchvision timm transformers scikit-learn matplotlib pillow tqdm umap-learn
```

If running locally, install dependencies manually:

```bash
pip install -r requirements.txt
```

---

# Running the Notebook

Run the notebook cells sequentially from top to bottom.

The notebook performs the following stages:

1. Dependency installation
2. Model loading (DINOv2)
3. Dataset loading
4. Embedding extraction
5. Similarity computation
6. Retrieval visualization
7. Evaluation metrics
8. Failure analysis visualizations

---

# Model Architecture

Feature extractor:
- `facebook/dinov2-base`

Embedding strategy:
- `[CLS]` token extraction
- L2 normalization

Similarity metric:
- cosine similarity

The pipeline uses pretrained embeddings rather than task-specific training to prioritize rapid prototyping, interpretability, and evaluation quality.

---

# Evaluation

The notebook evaluates:
- Rank-1 Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

It also generates:
- retrieval visualizations
- similarity score distributions
- embedding visualizations (t-SNE)
- qualitative failure case analysis

---

# Example Output

The notebook returns ranked retrieval results for a query image:

```text
Query Image:
dog_001/img3.jpg

Top Matches:
1. dog_001/img7.jpg — similarity 0.93
2. dog_001/img2.jpg — similarity 0.91
3. dog_004/img1.jpg — similarity 0.74
```

---

# Notes

- The notebook excludes the query image itself during retrieval ranking.
- A threshold-based same/different classification mode is also included.
- Open-set rejection can optionally be implemented by rejecting low-similarity queries.

---

# Limitations

Current limitations include:
- small curated evaluation dataset
- no fine-tuning on dog-specific ReID data
- no explicit dog detection or segmentation
- sensitivity to pose variation and background interference

These limitations are discussed further in `REPORT.md`.

---

# Future Improvements

Potential future extensions:
- metric-learning fine-tuning
- hard negative mining
- dog detection/cropping
- larger same-breed datasets
- open-set calibration
- local feature matching

---

# Acknowledgements

- DINOv2 pretrained model from Meta AI
- DogFaceNet dataset contributors
- Hugging Face Transformers
- PyTorch ecosystem
