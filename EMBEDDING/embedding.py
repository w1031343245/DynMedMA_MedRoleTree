import os
from sentence_transformers import SentenceTransformer
import numpy as np
from LABEL_EXTRACTION.data_clean import *

def embedding(word:str, embedding_dir: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = SentenceTransformer("/mnt/WR/LLM/bge-large-en-v1.5")
    vector = np.array(model.encode(word))
    clean_word = data_clean(word)
    np.save(f"{embedding_dir}/{clean_word}.npy", vector)

    return f"{embedding_dir}/{clean_word}.npy"