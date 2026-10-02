# Sutra Saree Retrieval

A PyTorch image-retrieval pipeline for Indian saree patterns. The project trains an embedding model, evaluates gallery/query retrieval, and supports inference on new saree images.

## Deliverables

- `notebooks/sutra_saree_retrieval.ipynb` — end-to-end Kaggle/Jupyter notebook
- Approach note covering architecture, preprocessing, training, sampling, augmentations, and losses
- Identification and verification evaluation with an explicit gallery/query split
- Efficiency reporting for embedding size, parameters, FLOPs, and inference latency

## Dataset

Use the [Indian Saree Patterns dataset on Kaggle](https://www.kaggle.com/datasets/div456/indian-sareepatterns). In Kaggle, attach the dataset to the notebook and update the dataset root cell if the mounted path differs.

The notebook also supports the supplied saree archive folders when copied into the runtime.

## Running the notebook

### Kaggle

1. Create a new Kaggle Notebook.
2. Upload `notebooks/sutra_saree_retrieval.ipynb`, or copy its cells into a notebook.
3. Add the Indian Saree Patterns dataset using **Add Input**.
4. Select a GPU accelerator.
5. Set the dataset path in the configuration cell.
6. Run every cell from top to bottom using **Run All**.

### Local Jupyter

```bash
pip install torch torchvision scikit-learn pandas numpy pillow matplotlib tqdm
jupyter notebook notebooks/sutra_saree_retrieval.ipynb
```

Set `DATA_ROOT` to the extracted dataset directory before running the training cells.

## Pipeline

1. Discover images and derive class labels from the dataset structure.
2. Remove unreadable files and apply deterministic train/validation/test splits.
3. Resize and normalize images using ImageNet-compatible preprocessing.
4. Apply training-only geometric and color augmentations.
5. Train a compact CNN embedding model with classification and metric-learning objectives.
6. Normalize embeddings and build a cosine-similarity gallery.
7. Evaluate top-k identification and verification metrics on held-out queries.
8. Export the trained checkpoint and use it for single-image or batch inference.

## Evaluation protocol

The notebook separates identities/classes into gallery and query samples so that queries are not used to construct their own reference representation. Identification is reported using Recall@1, Recall@5, and mean reciprocal rank. Verification is reported using cosine-similarity thresholds, ROC-AUC, and equal-error-rate-style analysis where enough positive and negative pairs are available.

Results printed by the notebook are the authoritative results for the selected split and seed. Dataset versions, class counts, split seed, and training configuration are displayed in the notebook output for reproducibility.

## Reproducibility

- Set the seed in the configuration cell.
- Keep the same dataset version and split strategy.
- Run all cells in order on the same device type when comparing results.
- Save the generated checkpoint and evaluation tables with the experiment outputs.

## Notes

The notebook is designed to run as-is in Kaggle with a GPU, but CPU execution is supported for debugging. Training time and retrieval quality depend on the number of images, class balance, image resolution, and available accelerator.

