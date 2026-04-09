import numpy as np

data = np.load('./amazon-giftcard/train_list.npy', allow_pickle=True)
data0 = np.load('./amazon-giftcard/valid_list.npy', allow_pickle=True)
data1 = np.load('./amazon-giftcard/test_list.npy', allow_pickle=True)

with open('datacheck.txt', 'w') as f:
    f.write(f"Train set data shape: {data.shape}\n")
    f.write(f"Train set data type: {data.dtype}\n")
    f.write(f"Train set data dimensions: {data.ndim}\n\n")

    f.write(f"Valid set data shape: {data0.shape}\n")
    f.write(f"Valid set data type: {data0.dtype}\n")
    f.write(f"Valid set data dimensions: {data0.ndim}\n\n")

    f.write(f"Test set data shape: {data1.shape}\n")
    f.write(f"Test set data type: {data1.dtype}\n")
    f.write(f"Test set data dimensions: {data1.ndim}\n\n")

    f.write("First 10 rows of train set data:\n")
    f.write(str(data[:10]))

    f.write(f"\n\nTrain set user ID range: {data[:, 0].min()} ~ {data[:, 0].max()}\n")
    f.write(f"Train set item ID range: {data[:, 1].min()} ~ {data[:, 1].max()}\n")
    f.write(f"Train set interaction count: {len(data)}\n")
    
    f.write(f"\n\nValid set user ID range: {data0[:, 0].min()} ~ {data0[:, 0].max()}\n")
    f.write(f"Valid set item ID range: {data0[:, 1].min()} ~ {data0[:, 1].max()}\n")
    f.write(f"Valid set interaction count: {len(data0)}\n")
    
    f.write(f"\n\nTest set user ID range: {data1[:, 0].min()} ~ {data1[:, 0].max()}\n")
    f.write(f"Test set item ID range: {data1[:, 1].min()} ~ {data1[:, 1].max()}\n")
    f.write(f"Test set interaction count: {len(data1)}\n")

    f.write(f"\nTotal interaction count: {len(data) + len(data0) + len(data1)}")