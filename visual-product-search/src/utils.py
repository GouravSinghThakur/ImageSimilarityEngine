"""
Faiss utility function
"""
from pathlib import Path
import faiss
import numpy as np

def load_index(index_path):
    """
    load the Saved Faiss Index
    """
    return faiss.read_index(str(index_path))

def search_index(index,query_embd,top_k=int()):
    """
    search Faiss index
    Args: 
        index: load Faiss index
        query_embd: shape(),embedding_dim)
        top_k: Number of Nearest Neighbours
    Returns:
        distances, indics

    """
    query_embd=query_embd.astype(np.float32)
    distances,indics=index.search(query_embd,top_k)
    return distances,indics