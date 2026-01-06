# preprocess_data_v0_fixed.py
import json
import ast  # 新增：用於解析Python字典格式
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# ===== 設定 =====
RAW_DIR = './raw_amazon_insturments'
PROCESSED_DIR = './amazon-instruments'
os.makedirs(PROCESSED_DIR, exist_ok=True)

REVIEW_FILE = os.path.join(RAW_DIR, 'reviews_Musical_Instruments_5.json')
META_FILE = os.path.join(RAW_DIR, 'meta_Musical_Instruments.json')

# ===== Step 1: 解析評論 (rating >= 4) =====
print("Loading reviews...")
interactions = []
with open(REVIEW_FILE, 'r') as f:
    for line in tqdm(f, desc="Reading reviews"):
        try:
            r = json.loads(line)
            if r['overall'] >= 4:  # 只保留评分>=4的交互
                interactions.append((r['reviewerID'], r['asin'], r['unixReviewTime']))
        except:
            continue

df = pd.DataFrame(interactions, columns=['user_id', 'asin', 'timestamp'])
print(f"Total clean interactions: {len(df)}")

# ===== Step 2: 按用戶和時間排序 =====
print("Sorting by user_id and timestamp...")
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ===== Step 3: 構建 ID 映射 =====
print("Constructing ID mapping...")
all_users = sorted(df['user_id'].unique())
all_items = sorted(df['asin'].unique())

user_map = {u: i for i, u in enumerate(all_users)}
item_map = {a: i for i, a in enumerate(all_items)}

with open(os.path.join(PROCESSED_DIR, 'user_map.pkl'), 'wb') as f:
    pickle.dump(user_map, f)
with open(os.path.join(PROCESSED_DIR, 'item_map.pkl'), 'wb') as f:
    pickle.dump(item_map, f)
print("ID mapping construction success.")

# ===== Step 4: 按用戶進行 7:2:1 切分 =====
print("Splitting per user with 7:2:1 ratio...")
train_pairs = []
valid_pairs = []
test_pairs = []

# 統計信息
train_interactions = 0
valid_interactions = 0
test_interactions = 0

for user_original, group in tqdm(df.groupby('user_id'), desc="Processing users"):
    user_mapped = user_map[user_original]
    
    # 按時間順序分組
    n = len(group)
    train_end = int(0.7 * n)
    valid_end = int(0.9 * n)
    
    # 確保每個用戶至少有1個訓練樣本和1個測試樣本
    if train_end == 0:
        train_end = 1
    if valid_end <= train_end:
        valid_end = train_end + 1 if n > train_end + 1 else train_end
    
    # 按時間順序收集交互
    for i, (_, row) in enumerate(group.iterrows()):
        item_mapped = item_map[row['asin']]
        pair = [user_mapped, item_mapped]
        
        if i < train_end:
            train_pairs.append(pair)
            train_interactions += 1
        elif i < valid_end:
            valid_pairs.append(pair)
            valid_interactions += 1
        else:
            test_pairs.append(pair)
            test_interactions += 1

print(f"Train interactions: {train_interactions}")
print(f"Valid interactions: {valid_interactions}")
print(f"Test interactions: {test_interactions}")

# ===== Step 4.5: 確保所有物品都在訓練集中出現 =====
print("Ensuring all items appear in training set...")

# 收集訓練集中出現的物品
train_items = set([iid for _, iid in train_pairs])

# 過濾驗證集和測試集
valid_pairs_filtered = []
test_pairs_filtered = []

removed_valid = 0
removed_test = 0

for uid, iid in valid_pairs:
    if iid in train_items:
        valid_pairs_filtered.append([uid, iid])
    else:
        removed_valid += 1

for uid, iid in test_pairs:
    if iid in train_items:
        test_pairs_filtered.append([uid, iid])
    else:
        removed_test += 1

valid_pairs = valid_pairs_filtered
test_pairs = test_pairs_filtered

print(f"Removed {removed_valid} items from validation set")
print(f"Removed {removed_test} items from test set")
print(f"Final counts: Train={len(train_pairs)}, Valid={len(valid_pairs)}, Test={len(test_pairs)}")

# ===== Step 5: 驗證排序和切分正確性 =====
print("\nVerifying split correctness...")

# 檢查每個用戶的交互是否連續
train_dict = {}
for uid, iid in train_pairs:
    if uid not in train_dict:
        train_dict[uid] = []
    train_dict[uid].append(iid)

# 檢查訓練集中每個用戶的交互是否按時間排序（實際上是按收集順序）
print(f"Number of users in train set: {len(train_dict)}")

# 檢查是否有用戶在訓練集中但不在驗證/測試集中
train_users = set([uid for uid, _ in train_pairs])
valid_users_set = set([uid for uid, _ in valid_pairs])
test_users_set = set([uid for uid, _ in test_pairs])

print(f"Users in train set: {len(train_users)}")
print(f"Users in valid set: {len(valid_users_set)}")
print(f"Users in test set: {len(test_users_set)}")

# 檢查是否有用戶在驗證/測試集中但不在訓練集中
missing_in_train = (valid_users_set | test_users_set) - train_users
if len(missing_in_train) > 0:
    print(f"WARNING: {len(missing_in_train)} users in valid/test but not in train")

# ===== Step 6: 輸出 .npy 文件 =====
print("\nSaving .npy files...")
def save_npy(pairs, path):
    np.save(path, np.array(pairs, dtype=np.int32))
    print(f"  Saved {len(pairs)} pairs to {path}")

save_npy(train_pairs, os.path.join(PROCESSED_DIR, 'train_list.npy'))
save_npy(valid_pairs, os.path.join(PROCESSED_DIR, 'valid_list.npy'))
save_npy(test_pairs, os.path.join(PROCESSED_DIR, 'test_list.npy'))

# 保存統計信息
stats = {
    'n_users': len(all_users),
    'n_items': len(all_items),
    'train_interactions': train_interactions,
    'valid_interactions': valid_interactions,
    'test_interactions': test_interactions,
    'total_interactions': len(df)
}
with open(os.path.join(PROCESSED_DIR, 'dataset_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

# ===== Step 7: 提取物品文本（修正版）=====
print("\nLoading metadata for item texts...")
item_texts = {}
if os.path.exists(META_FILE):
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                # 使用 ast.literal_eval 而不是 json.loads，因為文件是Python字典格式
                m = ast.literal_eval(line.strip())
                asin = m.get('asin')
                if asin not in item_map:
                    continue
                
                # 提取文本字段
                title = m.get('title', '').strip()
                
                # 處理 description（可能是字符串或列表）
                description = m.get('description', '')
                if isinstance(description, list):
                    desc = ' '.join([str(d).strip() for d in description if d]).strip()
                else:
                    desc = str(description).strip() if description else ''
                
                # 處理 categories（可能是 categories 或 category）
                categories = m.get('categories', m.get('category', []))
                if categories:
                    # 展平嵌套列表
                    flat_categories = []
                    for cat in categories:
                        if isinstance(cat, list):
                            flat_categories.extend([str(c).strip() for c in cat if c])
                        else:
                            flat_categories.append(str(cat).strip())
                    cats = ' '.join(flat_categories)
                else:
                    cats = ''
                
                # 組合文本
                text_parts = []
                if title:
                    text_parts.append(title)
                if desc:
                    text_parts.append(desc)
                if cats:
                    text_parts.append(cats)
                
                text = ' '.join(text_parts).strip()
                
                # 如果所有字段都為空，使用回退文本
                if not text:
                    text = f"instruments_{asin}"
                    
                item_texts[asin] = text
            except Exception as e:
                # 如果 ast.literal_eval 失敗，嘗試 json.loads
                try:
                    m = json.loads(line)
                    asin = m.get('asin')
                    if asin and asin in item_map:
                        # 提取文本（簡化版本）
                        title = m.get('title', '').strip()
                        text = title if title else f"instruments_{asin}"
                        item_texts[asin] = text
                except:
                    continue

    # 按 item_id 順序輸出文本列表
    print("Saving item text list...")
    reverse_item_map = {i: a for a, i in item_map.items()}
    num_items = len(item_map)
    
    item_text_list = []
    missing_count = 0
    
    for i in range(num_items):
        asin = reverse_item_map[i]
        if asin in item_texts:
            item_text_list.append(item_texts[asin])
        else:
            # 如果在 metadata 中找不到，使用回退文本
            item_text_list.append(f"instruments_{asin}")
            missing_count += 1
    
    print(f"Total items: {num_items}")
    print(f"Items with metadata: {num_items - missing_count}")
    print(f"Items without metadata (using fallback): {missing_count}")
    
    # 保存文本列表
    with open(os.path.join(PROCESSED_DIR, 'item_text_list.txt'), 'w', encoding='utf-8') as f:
        for text in item_text_list:
            f.write(text + '\n')
    
    print(f"Saved text for {len(item_text_list)} items")
    
    # 輸出示例以驗證
    print("\nFirst 10 item texts for verification:")
    for i in range(min(10, len(item_text_list))):
        asin = reverse_item_map[i]
        text = item_text_list[i]
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  Item {i} (ASIN: {asin}): {text_preview}")
        
else:
    print("Metadata file not found, skipping text extraction")

# ===== Step 8: 生成數據示例以驗證 =====
print("\nGenerating example for verification...")
# 選取前5個用戶展示他們的交互分配
sample_users = list(user_map.keys())[:5]
sample_file = os.path.join(PROCESSED_DIR, 'data_split_example.txt')
with open(sample_file, 'w') as f:
    f.write("Data Split Example (First 5 users):\n")
    f.write("=" * 60 + "\n")
    
    for user_original in sample_users:
        user_mapped = user_map[user_original]
        
        # 收集該用戶在所有集合中的交互
        user_train = [(uid, iid) for uid, iid in train_pairs if uid == user_mapped]
        user_valid = [(uid, iid) for uid, iid in valid_pairs if uid == user_mapped]
        user_test = [(uid, iid) for uid, iid in test_pairs if uid == user_mapped]
        
        f.write(f"\nUser {user_original} (mapped to {user_mapped}):\n")
        f.write(f"  Train: {len(user_train)} interactions (earliest)\n")
        f.write(f"  Valid: {len(user_valid)} interactions (middle)\n")
        f.write(f"  Test:  {len(user_test)} interactions (latest)\n")
        
        if len(user_train) > 0:
            f.write(f"  Train items: {[iid for _, iid in user_train[:5]]}")
            if len(user_train) > 5:
                f.write(f" ... and {len(user_train)-5} more\n")
            else:
                f.write("\n")

print("\n✅ Data preprocessing completed successfully!")
print(f"Output directory: {PROCESSED_DIR}")
print(f"\nDataset Statistics:")
print(f"  Users: {len(all_users)}")
print(f"  Items: {len(all_items)}")
print(f"  Total interactions: {len(df)}")
print(f"  Train interactions: {train_interactions} ({train_interactions/len(df)*100:.1f}%)")
print(f"  Valid interactions: {valid_interactions} ({valid_interactions/len(df)*100:.1f}%)")
print(f"  Test interactions: {test_interactions} ({test_interactions/len(df)*100:.1f}%)")