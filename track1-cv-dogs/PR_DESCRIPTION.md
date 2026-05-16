# Candidate Details

- Name: Najwa Ibrahimi
- Email: najwa.ibrahimi@utoronto.ca
- Track: Computer Vision — Dog Re-Identification

---

# Summary

Built a dog re-identification pipeline using pretrained DINOv2 embeddings and cosine similarity retrieval.

The system:
- Accepts a reference image of a dog
- Computes a visual embedding using DINOv2
- Retrieves and ranks the most visually similar query images
- Predicts whether query images contain the same dog

Key design decisions:
- Used pretrained DINOv2 embeddings instead of training a custom metric-learning model from scratch
- Used cosine similarity over L2-normalized embeddings for retrieval
- Curated a smaller identity-based evaluation dataset to prioritize evaluation quality and rapid experimentation
- Evaluated both retrieval performance and binary same/different classification behavior

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

# Approach Walkthrough

## Feature Extractor

Used `facebook/dinov2-base` as the embedding backbone.

DINOv2 was chosen because:
- it produces strong general-purpose visual embeddings
- it performs well on retrieval-style tasks
- it is self-supervised and optimized for visual representation quality
- it avoids the need for expensive task-specific training

Compared to CLIP, DINOv2 is more suitable for fine-grained visual matching because its embeddings are not optimized around image-text alignment or semantic category prediction.

---

## Embedding Strategy

For each image:
1. Load and preprocess image
2. Pass image through DINOv2
3. Extract the `[CLS]` token from the final transformer layer
4. Apply L2 normalization

This produces a fixed-length embedding vector for each image.

Similarity between images is computed using cosine similarity.

Because embeddings are normalized:
- cosine similarity
- dot product ranking

become equivalent.

---

## Retrieval Pipeline

Given a reference image:
1. Compute reference embedding
2. Compare against all query embeddings
3. Rank query images by cosine similarity
4. Return top matches and similarity scores

The system supports:
- ranked retrieval
- binary same/different classification using a similarity threshold

---

## Dataset Preparation

A curated subset of dog identities was used for evaluation.

Dataset structure:

```text
sample_dataset/
    dog_001/
    dog_002/
    dog_003/
```

Each folder corresponds to a single individual dog identity.

A smaller curated subset was intentionally used instead of the full dataset to:
- improve data quality
- reduce noisy labels
- enable rapid experimentation and debugging
- focus effort on evaluation and analysis

---

# Results

## Evaluation Metrics

| Metric | Value |
|---|---|
| Rank-1 Accuracy | 82.89% |
| Precision | 88.89% |
| Recall | 37.50% |
| F1 Score | 52.75% |
| ROC-AUC | 96.56% |

---

## Similarity Distribution

![Cosine Similarity](track1-cv-dogs/outputs/plots/cosine_similarity.png)

This plot shows separation between:
- same-dog similarity scores
- different-dog similarity scores

The overlap region highlights difficult retrieval cases and threshold sensitivity.

---

## Retrieval Examples

### Successful Matches


![Retrieval Examples](track1-cv-dogs/outputs/retrieval_examples/retrieval_example1.png)

Observed strengths:
- good robustness to lighting changes
- moderate robustness to pose variation
- strong retrieval when coat texture and facial features are visible

---

## Failure Cases

### Failure Case 1 — Similar Appearance Between Different Dogs

![Failure Case 1](track1-cv-dogs/outputs/failure_cases/failure_case_example4.png)

Dogs with:
- similar coat colors
- similar facial markings
- similar pose/background

sometimes produced highly similar embeddings despite being different identities.

Potential mitigation:
- train a domain-specific metric learning model
- add hard negative mining
- incorporate local feature matching

---

### Failure Case 2 — Pose and Occlusion Variation

![Failure Case 2](track1-cv-dogs/outputs/failure_cases/failure_case_example5.png)

Performance degraded when:
- the dog was partially occluded
- the face was not visible
- the viewing angle differed significantly from gallery images

Potential mitigation:
- dog detection and cropping before embedding
- multi-view gallery images
- augmentation during fine-tuning

---

# Open-Set Recognition (Optional Extension)

Implemented threshold-based rejection for unknown dogs.

If the maximum similarity score falls below a threshold:
- the query is classified as unknown

This prevents forced matches when the query dog does not appear in the gallery.

Potential future improvements:
- adaptive threshold calibration
- confidence estimation
- probabilistic retrieval scoring

---

# What Didn't Work

## Full Dataset Usage

Initial experiments using larger uncurated datasets introduced:
- inconsistent image quality
- noisy identity labels
- large viewpoint variation

This made debugging and evaluation more difficult.

A smaller curated dataset produced more interpretable results and faster iteration.

---

## Fixed Similarity Threshold

A single global threshold did not generalize perfectly across all dogs.

Some identities consistently produced:
- very high similarity scores
while others had:
- lower intra-class similarity

An adaptive or calibrated threshold would likely improve recall.

---

# Known Limitations

- Small-scale evaluation dataset
- No fine-tuning performed on dog-specific ReID data
- No explicit dog detection or segmentation
- Background information can influence embeddings
- Performance decreases under heavy occlusion or extreme viewpoint changes
- Same-breed dogs remain especially challenging

---

# Future Improvements

Potential next steps:
- Fine-tune using triplet loss or contrastive learning
- Add YOLO-based dog detection/cropping
- Use hard negative mining
- Evaluate on larger same-breed datasets
- Ensemble global and local visual features
- Add temporal tracking for video-based identification

---

# Example Outputs


## Embedding Visualization

![Embedding Visualization](track1-cv-dogs/outputs/plots/t-sne.png)

This visualization shows clustering behavior in embedding space across dog identities.

---

# Final Notes

The focus of this project was:
- building a clean and interpretable ReID prototype
- evaluating retrieval behavior rigorously
- understanding system failure modes

rather than optimizing for maximum benchmark performance or training a production-scale model from scratch.
