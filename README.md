# Image Similarity Engine

A visual product search system that finds visually similar products from an uploaded query image. The project uses a ResNet50 feature extractor to generate image embeddings and FAISS for fast nearest-neighbor similarity search, with an interactive Streamlit interface for testing retrieval results.

## Overview

This repository contains an end-to-end image similarity workflow:

- Dataset exploration and preprocessing notebooks
- ResNet50-based feature extraction
- Embedding generation for product images
- FAISS index creation and search
- Streamlit web app for uploading an image and viewing similar products

The current implementation is designed around the Stanford Online Products dataset and stores generated embeddings and FAISS indexes under the project `data/` directory.

## Features

- Image-to-image product retrieval
- ResNet50 ImageNet feature embeddings
- 2048-dimensional visual feature vectors
- FAISS-powered similarity search
- Cosine similarity through L2-normalized embeddings
- Streamlit web UI with configurable top-k results
- Notebook-based experimentation pipeline

## Project Structure

```text
.
├── LICENSE
├── README.md
└── visual-product-search/
    ├── app.py
    ├── main.py
    ├── pyproject.toml
    ├── requirements.txt
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb
    │   ├── 02_data_pipeline.ipynb
    │   ├── 03_feature_extraction.ipynb
    │   ├── 04_embedding_generation.ipynb
    │   ├── 05_similarity_search.ipynb
    │   ├── 06_Retrivel.ipynb
    │   └── 07_Evalution.ipynb
    └── src/
        ├── config.py
        ├── dataset.py
        ├── model.py
        ├── search.py
        ├── transforms.py
        └── utils.py
```

Generated data and model artifacts are intentionally ignored by Git:

```text
visual-product-search/data/raw/
visual-product-search/data/embeddings/
visual-product-search/data/faiss/
visual-product-search/models/
```

## Tech Stack

- Python
- PyTorch
- TorchVision
- FAISS
- Streamlit
- NumPy
- Pandas
- Pillow
- OpenCV
- Jupyter Notebook

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd image_Similarity/visual-product-search
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you use a CUDA-enabled GPU, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide.

## Dataset Setup

The project expects the Stanford Online Products dataset files under:

```text
visual-product-search/data/raw/
```

Expected paths from `src/config.py`:

```text
data/raw/Stanford_Online_Products/
data/raw/Ebay_train.txt
data/raw/Ebay_test.txt
```

After preparing the dataset, run the notebooks in order to generate embeddings and indexes.

## Workflow

Run the notebooks in this sequence:

1. `notebooks/01_dataset_exploration.ipynb`
2. `notebooks/02_data_pipeline.ipynb`
3. `notebooks/03_feature_extraction.ipynb`
4. `notebooks/04_embedding_generation.ipynb`
5. `notebooks/05_similarity_search.ipynb`
6. `notebooks/06_Retrivel.ipynb`
7. `notebooks/07_Evalution.ipynb`

The Streamlit app expects these generated artifacts:

```text
data/embeddings/train_embeddings.npy
data/embeddings/train_labels.npy
data/embeddings/train_paths.pkl
data/faiss/train_index_v1.faiss
```

## Running the App

From the `visual-product-search` directory:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, upload a product image, choose the number of results, and run the search.

## How It Works

1. The query image is resized and normalized using ImageNet preprocessing.
2. A pretrained ResNet50 model extracts a 2048-dimensional feature vector.
3. The vector is L2-normalized for cosine-style similarity search.
4. FAISS searches the prebuilt product embedding index.
5. The closest product images are displayed in the Streamlit interface.

## Configuration

Main configuration values are defined in:

```text
visual-product-search/src/config.py
```

Important settings:

- `IMG_SIZE = 128`
- `EMBD_DIM = 2048`
- `BATCH_SIZE = 8`
- `TRAIN_INDEX = data/faiss/train_index_v1.faiss`
- `TRAIN_PATHS = data/embeddings/train_paths.pkl`

## Troubleshooting

### FAISS index not found

Make sure the embedding and FAISS generation notebooks have been run successfully and that `data/faiss/train_index_v1.faiss` exists.

### Image paths fail to load

Check that `data/embeddings/train_paths.pkl` points to valid image files on your machine.

### Slow search or feature extraction

The app automatically uses CUDA when available. If it runs on CPU, search will still work, but feature extraction may be slower.

### Dependency installation issues

Install PyTorch separately if the default package resolution does not match your Python or CUDA version.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
