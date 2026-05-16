# Dog Re-Identification Report

## Approach

This project implements a prototype dog re-identification (ReID) pipeline using pretrained DINOv2 embeddings and cosine similarity retrieval. The goal of the system is to determine whether a query image contains the same individual dog as a provided reference image. Unlike breed classification, the task focuses on distinguishing between individual dogs, including dogs of the same breed.

The pipeline was designed around pretrained visual embeddings rather than training a custom model from scratch. Given the limited assessment timeline and relatively small evaluation dataset, leveraging a strong pretrained representation provided a more practical and interpretable approach. The feature extractor used was `facebook/dinov2-base`, a self-supervised vision transformer trained on large-scale image data. DINOv2 was selected because it produces high-quality visual embeddings that perform well on retrieval-style tasks without requiring task-specific fine-tuning.

For each image, the system extracts the `[CLS]` token from the final transformer layer and applies L2 normalization to produce a fixed-length embedding vector. Similarity between images is computed using cosine similarity. Query images are ranked according to similarity score, allowing the system to function both as a retrieval pipeline and as a binary same/different classifier using a similarity threshold.

A curated identity-based dataset structure was used, where each folder represented one individual dog identity with multiple images captured under varying conditions. This setup allowed evaluation on both positive matches (same dog) and negative matches (different dogs). Performance was evaluated using Rank-1 accuracy, precision, recall, F1 score, and similarity score distributions. Qualitative retrieval visualizations were also generated to analyze both successful matches and failure cases.

---

## Failure Modes

One observed failure mode involved dogs with very similar visual appearance. Dogs with similar coat colors, facial structure, or fur patterns sometimes produced highly similar embeddings despite belonging to different identities. This issue was especially noticeable for visually similar breeds. A potential mitigation would be fine-tuning the model using metric learning approaches such as triplet loss or contrastive learning with hard negative mining. This would encourage the embedding space to better separate visually similar individuals.

Another common failure mode involved pose variation and background interference. Performance degraded when query images contained side profiles, partial occlusions, low-resolution images, or visually dominant backgrounds. Since DINOv2 processes the full image, the embedding can be influenced by background scenery rather than the dog itself. A practical mitigation would be introducing a dog detection or segmentation stage prior to embedding extraction. Cropping the dog before feature extraction would reduce background influence and improve consistency across viewpoints.

---

## Generalisation to Other Species

Applying this pipeline to a species with significantly less publicly available data, such as sheep, would introduce several challenges. The current approach assumes the availability of multiple labeled images per identity and sufficient visual distinctiveness between individuals. For less-studied species, both assumptions may fail.

First, there may be very limited identity-labeled data available. In that setting, collecting even a small curated dataset would become a major bottleneck. Fine-tuning large models would also become more difficult due to overfitting risks. Few-shot or self-supervised adaptation techniques would likely become more important.

Second, the visual cues used for identification may differ substantially between species. Dogs often exhibit strong variation in facial features, coat texture, and color patterns, which pretrained vision models can leverage effectively. Sheep, by comparison, may have lower inter-individual visual variability, making identity separation more difficult. In that case, the embedding model may require domain-specific fine-tuning or the incorporation of additional modalities such as temporal tracking, body shape analysis, or biometric markers.

Finally, the current pipeline assumes that pretrained representations learned on general internet-scale image data transfer reasonably well to dogs. This assumption may weaken for less common species with limited representation in pretraining datasets, reducing embedding quality and retrieval reliability.
