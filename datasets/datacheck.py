import numpy as np

# 讀取 .npy 文件
# data = np.load('DiffRec/amazon_item_emb/amazon-book_clean/item_emb.npy', allow_pickle=True)
# data = np.load('./amazon-book_mine/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-book_mine/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-book_mine/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-book_clean/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-book_clean/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-book_clean/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-book_noisy/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-book_noisy/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-book_noisy/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-book_small/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-book_small/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-book_small/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-movietv_small/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-movietv_small/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-movietv_small/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-instruments/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-instruments/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments_coldstart/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-instruments_coldstart/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-instruments_coldstart/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments_coldstart/embeddings/item_embeddings.npy', allow_pickle=True)

# data = np.load('./ml-1m_clean/train_list.npy', allow_pickle=True)
# data0 = np.load('./ml-1m_clean/valid_list.npy', allow_pickle=True)
# data1 = np.load('./ml-1m_clean/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments/embeddings/item_embeddings.npy', allow_pickle=True)

data = np.load('./amazon-Software/train_list.npy', allow_pickle=True)
data0 = np.load('./amazon-Software/valid_list.npy', allow_pickle=True)
data1 = np.load('./amazon-Software/test_list.npy', allow_pickle=True)


# # 查看基本信息
# print(f"數據形狀: {data.shape}")
# print(f"數據類型: {data.dtype}")
# print(f"數據維度: {data.ndim}")

# # 查看前幾行
# print("前50行數據:")
# print(data[:50])

# # 查看統計信息
# print(f"用戶ID範圍: {data[:, 0].min()} ~ {data[:, 0].max()}")
# print(f"物品ID範圍: {data[:, 1].min()} ~ {data[:, 1].max()}")
# print(f"交互數: {len(data)}")

# print(f"總交互數: {len(data) + len(data0) + len(data1)}")

with open('datacheck.txt', 'w') as f:
    f.write(f"訓練集數據形狀: {data.shape}\n")
    f.write(f"訓練集數據類型: {data.dtype}\n")
    f.write(f"訓練集數據維度: {data.ndim}\n\n")

    f.write(f"驗證集數據形狀: {data0.shape}\n")
    f.write(f"驗證集數據類型: {data0.dtype}\n")
    f.write(f"驗證集數據維度: {data0.ndim}\n\n")

    f.write(f"測試集數據形狀: {data1.shape}\n")
    f.write(f"測試集數據類型: {data1.dtype}\n")
    f.write(f"測試集數據維度: {data1.ndim}\n\n")

    f.write("訓練集前10行數據:\n")
    f.write(str(data[:10]))

    f.write(f"\n\n訓練集用戶ID範圍: {data[:, 0].min()} ~ {data[:, 0].max()}\n")
    f.write(f"訓練集物品ID範圍: {data[:, 1].min()} ~ {data[:, 1].max()}\n")
    f.write(f"訓練集交互數: {len(data)}\n")
    
    f.write(f"\n\n驗證集用戶ID範圍: {data0[:, 0].min()} ~ {data0[:, 0].max()}\n")
    f.write(f"驗證集物品ID範圍: {data0[:, 1].min()} ~ {data0[:, 1].max()}\n")
    f.write(f"驗證集交互數: {len(data0)}\n")
    
    f.write(f"\n\n測試集用戶ID範圍: {data1[:, 0].min()} ~ {data1[:, 0].max()}\n")
    f.write(f"測試集物品ID範圍: {data1[:, 1].min()} ~ {data1[:, 1].max()}\n")
    f.write(f"測試集交互數: {len(data1)}\n")

    f.write(f"\n總交互數: {len(data) + len(data0) + len(data1)}")