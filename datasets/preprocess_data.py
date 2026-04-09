import json
import ast
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# ===== Configuration =====
RAW_DIR = './raw_amazon_giftcard'
PROCESSED_DIR = './amazon-giftcard'
os.makedirs(PROCESSED_DIR, exist_ok=True)

REVIEW_FILE = os.path.join(RAW_DIR, 'Gift_Cards_5.json')
META_FILE = os.path.join(RAW_DIR, 'meta_Gift_Cards.json')

# ===== Step 1: Parse reviews (rating >= 4) =====
print("Loading reviews...")
interactions = []
with open(REVIEW_FILE, 'r') as f:
    for line in tqdm(f, desc="Reading reviews"):
        try:
            r = json.loads(line)
            if r['overall'] >= 4:  # Only keep interactions with rating >= 4
                interactions.append((r['reviewerID'], r['asin'], r['unixReviewTime']))
        except:
            continue

df = pd.DataFrame(interactions, columns=['user_id', 'asin', 'timestamp'])
print(f"Total clean interactions: {len(df)}")

# ===== Step 2: Sort by user and timestamp =====
print("Sorting by user_id and timestamp...")
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ===== Step 3: Build ID mapping =====
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

# ===== Step 4: Split 7:2:1 per user =====
print("Splitting per user with 7:2:1 ratio...")
train_pairs = []
valid_pairs = []
test_pairs = []

# Statistics
train_interactions = 0
valid_interactions = 0
test_interactions = 0

for user_original, group in tqdm(df.groupby('user_id'), desc="Processing users"):
    user_mapped = user_map[user_original]
    
    # Group by chronological order
    n = len(group)
    train_end = int(0.7 * n)
    valid_end = int(0.9 * n)
    
    # Ensure each user has at least 1 training and 1 test sample
    if train_end == 0:
        train_end = 1
    if valid_end <= train_end:
        valid_end = train_end + 1 if n > train_end + 1 else train_end
    
    # Collect interactions in chronological order
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

# ===== Step 4.5: Ensure all items appear in the training set (Refactored, moved to train instead of deleting) =====
print("Ensuring all items appear in training set (refactored)...")

# Items currently appearing in the training set
train_items = set([iid for _, iid in train_pairs])

# All items (from total df/item_map)
all_item_ids = set(item_map.values())

# Find items not appearing in train
missing_items = all_item_ids - train_items
print(f"Items not in initial train: {len(missing_items)}")

if len(missing_items) > 0:
    new_train_pairs = train_pairs[:]
    new_valid_pairs = []
    new_test_pairs = []

    moved_items = set()

    # Find the earliest interaction of these items in valid to move to train first
    for uid, iid in valid_pairs:
        if iid in missing_items and iid not in moved_items:
            new_train_pairs.append([uid, iid])
            moved_items.add(iid)
        else:
            new_valid_pairs.append([uid, iid])

    # Then find the remaining items in test (not found in valid)
    for uid, iid in test_pairs:
        if iid in missing_items and iid not in moved_items:
            new_train_pairs.append([uid, iid])
            moved_items.add(iid)
        else:
            new_test_pairs.append([uid, iid])

    print(f"Moved {len(moved_items)} items' interactions from valid/test to train.")

    train_pairs = new_train_pairs
    valid_pairs = new_valid_pairs
    test_pairs = new_test_pairs

# train should now cover all items
train_items = set([iid for _, iid in train_pairs])
missing_items_after = all_item_ids - train_items
print(f"Items still missing from train after fix: {len(missing_items_after)}")
if len(missing_items_after) > 0:
    print("WARNING: some items still do not appear in train (might only have bad data or extreme cases that were deleted)")
else:
    print("All items now appear at least once in the training set.")


# ===== Step 5: Verify sorting and splitting correctness =====
print("\nVerifying split correctness...")

# Check if each user's interactions are continuous
train_dict = {}
for uid, iid in train_pairs:
    if uid not in train_dict:
        train_dict[uid] = []
    train_dict[uid].append(iid)

# Check if each user's interactions in the training set are chronologically sorted (actually by collection order)
print(f"Number of users in train set: {len(train_dict)}")

# Check if there are users in train but not in valid/test
train_users = set([uid for uid, _ in train_pairs])
valid_users_set = set([uid for uid, _ in valid_pairs])
test_users_set = set([uid for uid, _ in test_pairs])

print(f"Users in train set: {len(train_users)}")
print(f"Users in valid set: {len(valid_users_set)}")
print(f"Users in test set: {len(test_users_set)}")

# Check if there are users in valid/test but not in train
missing_in_train = (valid_users_set | test_users_set) - train_users
if len(missing_in_train) > 0:
    print(f"WARNING: {len(missing_in_train)} users in valid/test but not in train")

# ===== Step 6: Output .npy files =====
print("\nSaving .npy files...")
def save_npy(pairs, path):
    np.save(path, np.array(pairs, dtype=np.int32))
    print(f"  Saved {len(pairs)} pairs to {path}")

save_npy(train_pairs, os.path.join(PROCESSED_DIR, 'train_list.npy'))
save_npy(valid_pairs, os.path.join(PROCESSED_DIR, 'valid_list.npy'))
save_npy(test_pairs, os.path.join(PROCESSED_DIR, 'test_list.npy'))

# Save statistics
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

# Modify Step 7 part in preprocess_data_v1.py

# ===== Step 7: Extract item texts (fixed version) =====
print("\nLoading metadata for item texts...")
item_texts = {}
if os.path.exists(META_FILE):
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                # Use ast.literal_eval instead of json.loads since the file is in Python dict format
                m = ast.literal_eval(line.strip())
                asin = m.get('asin')
                if asin not in item_map:
                    continue
                
                # Extract text fields
                title = m.get('title', '').strip()
                
                # Process description (could be string or list)
                description = m.get('description', '')
                if isinstance(description, list):
                    desc = ' '.join([str(d).strip() for d in description if d]).strip()
                else:
                    desc = str(description).strip() if description else ''
                
                # Process categories (could be categories or category)
                categories = m.get('categories', m.get('category', []))
                if categories:
                    # Flatten nested lists
                    flat_categories = []
                    for cat in categories:
                        if isinstance(cat, list):
                            flat_categories.extend([str(c).strip() for c in cat if c])
                        else:
                            flat_categories.append(str(cat).strip())
                    cats = ' '.join(flat_categories)
                else:
                    cats = ''
                
                # Combine text
                text_parts = []
                if title:
                    text_parts.append(title)
                if desc:
                    text_parts.append(desc)
                if cats:
                    text_parts.append(cats)
                
                text = ' '.join(text_parts).strip()
                
                # ===== Crucial modification: clear newlines in text =====
                # Replace all newlines with spaces, and merge extra spaces
                if text:
                    # Replace various newlines
                    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    # Merge multiple consecutive spaces into a single space
                    import re
                    text = re.sub(r'\s+', ' ', text).strip()
                
                # If all fields are empty, use fallback text
                if not text:
                    text = f"instruments_{asin}"
                    
                item_texts[asin] = text
            except Exception as e:
                # If ast.literal_eval fails, try json.loads
                try:
                    m = json.loads(line)
                    asin = m.get('asin')
                    if asin and asin in item_map:
                        # Extract text (simplified version)
                        title = m.get('title', '').strip()
                        text = title if title else f"instruments_{asin}"
                        # Clean newlines as well
                        if text:
                            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                            import re
                            text = re.sub(r'\s+', ' ', text).strip()
                        item_texts[asin] = text
                except:
                    continue

    # Output text list in item_id order
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
            # If not found in metadata, use fallback text
            item_text_list.append(f"instruments_{asin}")
            missing_count += 1
    
    print(f"Total items: {num_items}")
    print(f"Items with metadata: {num_items - missing_count}")
    print(f"Items without metadata (using fallback): {missing_count}")
    
    # ===== Crucial modification: verify no newlines in each line of text =====
    print("Verifying text format...")
    for i, text in enumerate(item_text_list):
        if '\n' in text or '\r' in text:
            print(f"Warning: Item {i} contains newline characters, cleaning...")
            item_text_list[i] = text.replace('\n', ' ').replace('\r', ' ').strip()
    
    # Save text list
    with open(os.path.join(PROCESSED_DIR, 'item_text_list.txt'), 'w', encoding='utf-8') as f:
        for text in item_text_list:
            f.write(text + '\n')
    
    print(f"Saved text for {len(item_text_list)} items")
    
    # Output example for verification
    print("\nFirst 10 item texts for verification:")
    for i in range(min(10, len(item_text_list))):
        asin = reverse_item_map[i]
        text = item_text_list[i]
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  Item {i} (ASIN: {asin}): {text_preview}")
        
else:
    print("Metadata file not found, skipping text extraction")

# ===== Step 8: Generate data example for verification =====
print("\nGenerating example for verification...")
# Select top 5 users to exhibit their interaction distribution
sample_users = list(user_map.keys())[:5]
sample_file = os.path.join(PROCESSED_DIR, 'data_split_example.txt')
with open(sample_file, 'w') as f:
    f.write("Data Split Example (First 5 users):\n")
    f.write("=" * 60 + "\n")
    
    for user_original in sample_users:
        user_mapped = user_map[user_original]
        
        # Collect this user's interactions across all sets
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

print("\n Data preprocessing completed successfully!")
print(f"Output directory: {PROCESSED_DIR}")
print(f"\nDataset Statistics:")
print(f"  Users: {len(all_users)}")
print(f"  Items: {len(all_items)}")
print(f"  Total interactions: {len(df)}")
print(f"  Train interactions: {train_interactions} ({train_interactions/len(df)*100:.1f}%)")
print(f"  Valid interactions: {valid_interactions} ({valid_interactions/len(df)*100:.1f}%)")
print(f"  Test interactions: {test_interactions} ({test_interactions/len(df)*100:.1f}%)")