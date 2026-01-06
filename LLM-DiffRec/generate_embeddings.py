import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ===== 設定 =====
PROCESSED_DIR = '../datasets/amazon-book_mine'
EMBEDDING_DIR = '../datasets/amazon-book_mine/embeddings'
MODEL_CACHE_DIR = '../models/bge-m3'

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

TEXT_FILE = os.path.join(PROCESSED_DIR, 'item_text_list.txt')
OUTPUT_EMB = os.path.join(EMBEDDING_DIR, 'item_embeddings.npy')

# ===== 讀取文本列表 =====
print("Loading item texts...")
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    item_text_list = [line.strip() for line in f]

print(f"Total items: {len(item_text_list)}")

# ===== 載入嵌入模型（可替換這裡！）=====
print("Loading embedding model...")
# 可替換模型示例：
# - 'BAAI/bge-m3'（當前）
# - 'GanymedeNil/text2vec-large-chinese'
# - 'intfloat/e5-mistral-7b-instruct'（需注意授權）
model = SentenceTransformer(
    'BAAI/bge-m3',
    cache_folder=MODEL_CACHE_DIR
)

# ===== 生成嵌入 =====
print("Generating embeddings...")
embeddings = model.encode(
    item_text_list,
    batch_size=128,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ===== 保存 =====
np.save(OUTPUT_EMB, embeddings)
print(f"✅ Embeddings saved to {OUTPUT_EMB}")
print(f"Shape: {embeddings.shape}")