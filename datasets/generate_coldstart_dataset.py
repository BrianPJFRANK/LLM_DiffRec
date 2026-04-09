# generate_coldstart_dataset.py
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm

def generate_coldstart_dataset(original_dir, coldstart_dir, cold_start_ratio=0.3):

    os.makedirs(coldstart_dir, exist_ok=True)

    print("Loading original data...")
    
    # Load mapping
    with open(os.path.join(original_dir, 'user_map.pkl'), 'rb') as f:
        user_map = pickle.load(f)
    
    with open(os.path.join(original_dir, 'item_map.pkl'), 'rb') as f:
        item_map = pickle.load(f)
    
    # Load interaction list
    train_pairs = np.load(os.path.join(original_dir, 'train_list.npy'))
    valid_pairs = np.load(os.path.join(original_dir, 'valid_list.npy'))
    test_pairs = np.load(os.path.join(original_dir, 'test_list.npy'))
    
    # Load item text
    with open(os.path.join(original_dir, 'item_text_list.txt'), 'r', encoding='utf-8') as f:
        item_texts = [line.strip() for line in f]
    
    # Load embeddings (if exist)
    embeddings_path = os.path.join(original_dir, 'embeddings', 'item_embeddings.npy')
    if os.path.exists(embeddings_path):
        item_embeddings = np.load(embeddings_path)
    else:
        item_embeddings = None
    
    print(f"Original dataset stats:")
    print(f"  Users: {len(user_map)}")
    print(f"  Items: {len(item_map)}")
    print(f"  Train interactions: {len(train_pairs)}")
    print(f"  Valid interactions: {len(valid_pairs)}")
    print(f"  Test interactions: {len(test_pairs)}")
    
    # ===== 2. Identify items in the training set =====
    print("\nIdentifying cold-start items...")
    
    # Items appearing in the training set
    train_items = set(train_pairs[:, 1])
    print(f"Items in training set: {len(train_items)}")
    
    # Items appearing in the test set
    test_items = set(test_pairs[:, 1])
    print(f"Items in test set: {len(test_items)}")
    
    # Cold-start items: in the test set but not in the training set
    cold_start_items = list(test_items - train_items)
    
    # If too few cold-start items, randomly select some test items as cold-start
    if len(cold_start_items) < len(test_items) * cold_start_ratio:
        # Calculate the number of cold-start items to add
        target_count = int(len(test_items) * cold_start_ratio)
        additional_needed = target_count - len(cold_start_items)
        
        if additional_needed > 0:
            # Randomly select from other items in the test set
            other_test_items = list(test_items - set(cold_start_items))
            if additional_needed <= len(other_test_items):
                additional_cold = np.random.choice(other_test_items, additional_needed, replace=False)
                cold_start_items.extend(additional_cold)
                cold_start_items = list(set(cold_start_items))  # Deduplicate
    
    print(f"Cold-start items identified: {len(cold_start_items)}")
    print(f"Cold-start ratio: {len(cold_start_items)/len(test_items):.2%}")
    
    # ===== 3. Filter cold-start test set =====
    print("\nFiltering cold-start test set...")
    
    # Test set: only keep interactions containing cold-start items
    cold_start_test_mask = np.isin(test_pairs[:, 1], cold_start_items)
    cold_start_test_pairs = test_pairs[cold_start_test_mask]
    
    print(f"Original test interactions: {len(test_pairs)}")
    print(f"Cold-start test interactions: {len(cold_start_test_pairs)}")
    
    # ===== 4. Update training and validation sets =====
    # Ensure training and validation sets do not contain cold-start items
    print("\nUpdating training and validation sets...")
    
    # Training set: remove interactions containing cold-start items (though theoretically there should be none)
    train_mask = ~np.isin(train_pairs[:, 1], cold_start_items)
    new_train_pairs = train_pairs[train_mask]
    
    # Validation set: remove interactions containing cold-start items
    valid_mask = ~np.isin(valid_pairs[:, 1], cold_start_items)
    new_valid_pairs = valid_pairs[valid_mask]
    
    print(f"Original train interactions: {len(train_pairs)}")
    print(f"New train interactions: {len(new_train_pairs)}")
    print(f"Removed from train: {len(train_pairs) - len(new_train_pairs)}")
    
    print(f"Original valid interactions: {len(valid_pairs)}")
    print(f"New valid interactions: {len(new_valid_pairs)}")
    print(f"Removed from valid: {len(valid_pairs) - len(new_valid_pairs)}")
    
    # ===== 5. Update item mapping =====
    print("\nUpdating item mapping...")
    
    # Find all occurring items (train+valid+cold-start test)
    all_items_in_use = set(
        list(new_train_pairs[:, 1]) + 
        list(new_valid_pairs[:, 1]) + 
        list(cold_start_test_pairs[:, 1])
    )
    
    # Create a new item mapping
    new_item_map = {}
    reverse_original_item_map = {v: k for k, v in item_map.items()}
    
    for new_idx, old_item_id in enumerate(sorted(all_items_in_use)):
        original_asin = reverse_original_item_map[old_item_id]
        new_item_map[original_asin] = new_idx
    
    print(f"Original items: {len(item_map)}")
    print(f"New items after cold-start filtering: {len(new_item_map)}")
    print(f"Items removed: {len(item_map) - len(new_item_map)}")
    
    # ===== 6. Remap all interactions =====
    print("\nRemapping interactions...")
    
    def remap_pairs(pairs, item_map_old_to_new):
        """Remap item IDs in the interaction pairs"""
        remapped = []
        for uid, old_iid in pairs:
            original_asin = reverse_original_item_map[old_iid]
            if original_asin in new_item_map:  # Only keep existing items
                new_iid = new_item_map[original_asin]
                remapped.append([uid, new_iid])
        return np.array(remapped, dtype=np.int32)
    
    # Create mapping from old item IDs to new item IDs
    old_to_new_item_map = {}
    for original_asin, new_iid in new_item_map.items():
        old_iid = item_map[original_asin]
        old_to_new_item_map[old_iid] = new_iid
    
    new_train_pairs_remapped = remap_pairs(new_train_pairs, old_to_new_item_map)
    new_valid_pairs_remapped = remap_pairs(new_valid_pairs, old_to_new_item_map)
    new_test_pairs_remapped = remap_pairs(cold_start_test_pairs, old_to_new_item_map)
    
    # ===== 7. Update item text list =====
    print("\nUpdating item text list...")
    
    # Create new item text list
    new_item_texts = []
    for original_asin, new_iid in sorted(new_item_map.items(), key=lambda x: x[1]):
        old_iid = item_map[original_asin]
        new_item_texts.append(item_texts[old_iid])
    
    # ===== 8. Update item embeddings (if exist) =====
    new_item_embeddings = None
    if item_embeddings is not None:
        print("Updating item embeddings...")
        new_item_embeddings = []
        for original_asin, new_iid in sorted(new_item_map.items(), key=lambda x: x[1]):
            old_iid = item_map[original_asin]
            new_item_embeddings.append(item_embeddings[old_iid])
        new_item_embeddings = np.array(new_item_embeddings)
    
    # ===== 9. Save new dataset =====
    print(f"\nSaving cold-start dataset to: {coldstart_dir}")
    
    print("Splitting cold-start test set 50/50 into Valid and Test...")
    np.random.seed(42)  # Fixed random seed to ensure reproducibility
    total_cold_interactions = len(new_test_pairs_remapped)
    shuffled_indices = np.random.permutation(total_cold_interactions)
    shuffled_cold_pairs = new_test_pairs_remapped[shuffled_indices]

    split_idx = total_cold_interactions // 2
    final_valid_cold_pairs = shuffled_cold_pairs[:split_idx]
    final_test_cold_pairs = shuffled_cold_pairs[split_idx:]
    
    # Save interaction list
    np.save(os.path.join(coldstart_dir, 'train_list.npy'), new_train_pairs_remapped)
    np.save(os.path.join(coldstart_dir, 'valid_list.npy'), final_valid_cold_pairs)
    np.save(os.path.join(coldstart_dir, 'test_list.npy'), final_test_cold_pairs)
    
    # Save mapping
    with open(os.path.join(coldstart_dir, 'user_map.pkl'), 'wb') as f:
        pickle.dump(user_map, f)  # User mapping remains unchanged
    
    with open(os.path.join(coldstart_dir, 'item_map.pkl'), 'wb') as f:
        pickle.dump(new_item_map, f)
    
    # Save item text
    with open(os.path.join(coldstart_dir, 'item_text_list.txt'), 'w', encoding='utf-8') as f:
        for text in new_item_texts:
            f.write(text + '\n')
    
    # Save item embeddings
    if new_item_embeddings is not None:
        embeddings_dir = os.path.join(coldstart_dir, 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, 'item_embeddings.npy'), new_item_embeddings)
    
    # ===== 10. Copy other potentially required files =====
    # Copy dataset_stats.json (if exists)
    stats_path = os.path.join(original_dir, 'dataset_stats.json')
    if os.path.exists(stats_path):
        shutil.copy2(stats_path, os.path.join(coldstart_dir, 'dataset_stats.json'))
    
    # ===== 11. Generate cold-start dataset statistics =====
    print("\n" + "="*50)
    print("COLD-START DATASET STATISTICS")
    print("="*50)
    
    # Calculate proportion of cold-start items in the test set
    test_items_cold = set(new_test_pairs_remapped[:, 1])
    all_items_in_test = len(test_items_cold)
    
    # Find test items not appearing in the training set (true cold-start)
    train_items_set = set(new_train_pairs_remapped[:, 1])
    true_cold_items = test_items_cold - train_items_set
    
    print(f"Users: {len(user_map)}")
    print(f"Items (total): {len(new_item_map)}")
    print(f"Items in training set: {len(train_items_set)}")
    print(f"Items in test set: {all_items_in_test}")
    print(f"True cold-start items in test set: {len(true_cold_items)}")
    print(f"Cold-start ratio (items): {len(true_cold_items)/all_items_in_test:.2%}")
    print(f"\nInteractions:")
    print(f"  Train: {len(new_train_pairs_remapped)}")
    print(f"  Valid: {len(final_valid_cold_pairs)}")
    print(f"  Test (cold-start): {len(final_test_cold_pairs)}")
    
    # Save detailed statistics
    cold_stats = {
        'n_users': len(user_map),
        'n_items': len(new_item_map),
        'n_items_train': len(train_items_set),
        'n_items_test': all_items_in_test,
        'n_cold_start_items': len(true_cold_items),
        'cold_start_ratio': len(true_cold_items) / all_items_in_test,
        'train_interactions': len(new_train_pairs_remapped),
        'valid_interactions': len(final_valid_cold_pairs),
        'test_interactions': len(final_test_cold_pairs),
        'total_cold_interaction': total_cold_interactions
    }
    
    with open(os.path.join(coldstart_dir, 'cold_start_stats.json'), 'w') as f:
        import json
        json.dump(cold_stats, f, indent=2)
    
    print(f"\nCold-start dataset generated successfully!")
    print(f"   Location: {coldstart_dir}")
    
    return cold_stats

if __name__ == "__main__":
    # Configuration parameters
    ORIGINAL_DIR = "../datasets/amazon-giftcard"
    COLDSTART_DIR = "../datasets/amazon-giftcard_coldstart"
    COLD_START_RATIO = 0.3  # 30% of test items are cold-start items
    
    # Generate cold-start dataset
    stats = generate_coldstart_dataset(ORIGINAL_DIR, COLDSTART_DIR, COLD_START_RATIO)
    
    # Output example for verification
    print("\n" + "="*50)
    print("VERIFICATION SAMPLE")
    print("="*50)
    
    # Load some data for verification
    test_pairs = np.load(os.path.join(COLDSTART_DIR, 'test_list.npy'))
    with open(os.path.join(COLDSTART_DIR, 'item_map.pkl'), 'rb') as f:
        item_map_cold = pickle.load(f)
    
    # Take first 5 test samples
    print("\nFirst 5 test interactions (cold-start):")
    for i in range(min(5, len(test_pairs))):
        uid, iid = test_pairs[i]
        # Get original ASIN
        reverse_map = {v: k for k, v in item_map_cold.items()}
        asin = reverse_map[iid]
        print(f"  User {uid} -> Item {iid} (ASIN: {asin})")