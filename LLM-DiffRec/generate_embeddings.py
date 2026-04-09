import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DIR = '../datasets/amazon-giftcard'
EMBEDDING_DIR = '../datasets/amazon-giftcard/embeddings'
MODEL_CACHE_DIR = '../models/bge-m3'

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

TEXT_FILE = os.path.join(PROCESSED_DIR, 'item_text_list.txt')
OUTPUT_EMB = os.path.join(EMBEDDING_DIR, 'item_embeddings.npy')

print("Loading item texts...")
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    item_text_list = [line.strip() for line in f]

print(f"Total items: {len(item_text_list)}")

print("Loading embedding model...")

model = SentenceTransformer(
    'BAAI/bge-m3',
    cache_folder=MODEL_CACHE_DIR
)
model.max_seq_length = 512

print("Generating embeddings...")
embeddings = model.encode(
    item_text_list,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save(OUTPUT_EMB, embeddings)
print(f"Embeddings saved to {OUTPUT_EMB}")
print(f"Shape: {embeddings.shape}")