from pathlib import Path
# project Paths
PROJECT_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_DIR/"data"
RAW_DIR=DATA_DIR/"raw"
EMBEDDINGS_DIR=DATA_DIR/"embeddings"
FAISS_DIR=DATA_DIR/"faiss"
MODEL_DIR=PROJECT_DIR/"models"

# DATASET
DATASET_PATH=RAW_DIR/"Stanford_Online_Products"
TEST_CSV=RAW_DIR/"Ebay_test.txt"
TRAIN_CSV=RAW_DIR/"Ebay_train.txt"

# Embeddings
TRAIN_EMBEDDINGS = EMBEDDINGS_DIR / "train_embeddings.npy"
TRAIN_LABELS = EMBEDDINGS_DIR/ "train_labels.npy"
TRAIN_PATHS = EMBEDDINGS_DIR / "train_paths.pkl"
TEST_EMBEDDINGS = EMBEDDINGS_DIR/ "test_embdings.npy"
TEST_LABELS = EMBEDDINGS_DIR / "test_labels.npy"
TEST_PATHS = EMBEDDINGS_DIR / "test_paths.pkl"

# FAISS
TRAIN_INDEX=FAISS_DIR/"train_index_v1.faiss"
TEST_INDEX=FAISS_DIR/"tes_index.faiss"

# MODEL
IMG_SIZE=128
EMBD_DIM=2048
BATCH_SIZE=8
