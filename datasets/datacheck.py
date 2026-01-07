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

data = np.load('./amazon-instruments/train_list.npy', allow_pickle=True)
data1 = np.load('./amazon-instruments/valid_list.npy', allow_pickle=True)
data0 = np.load('./amazon-instruments/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments_coldstart/train_list.npy', allow_pickle=True)
# data0 = np.load('./amazon-instruments_coldstart/valid_list.npy', allow_pickle=True)
# data1 = np.load('./amazon-instruments_coldstart/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments_coldstart/embeddings/item_embeddings.npy', allow_pickle=True)

# data = np.load('./ml-1m_clean/train_list.npy', allow_pickle=True)
# data0 = np.load('./ml-1m_clean/valid_list.npy', allow_pickle=True)
# data1 = np.load('./ml-1m_clean/test_list.npy', allow_pickle=True)

# data = np.load('./amazon-instruments/embeddings/item_embeddings.npy', allow_pickle=True)

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
    f.write(f"數據形狀: {data.shape}\n")
    f.write(f"數據類型: {data.dtype}\n")
    f.write(f"數據維度: {data.ndim}\n")

    f.write("前50行數據:\n")
    f.write(str(data[:500]))

    f.write(f"\n用戶ID範圍: {data[:, 0].min()} ~ {data[:, 0].max()}\n")
    f.write(f"物品ID範圍: {data[:, 1].min()} ~ {data[:, 1].max()}\n")
    f.write(f"交互數: {len(data)}\n")

    # f.write(f"總交互數: {len(data) + len(data0) + len(data1)}")