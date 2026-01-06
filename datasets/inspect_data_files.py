# inspect_metadata_keys.py
import ast
import json
import os
from collections import Counter

def inspect_metadata_keys(file_path, sample_lines=1000):
    """檢查metadata文件中的鍵"""
    print(f"檢查文件: {file_path}")
    print(f"文件存在: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        return
    
    all_keys = Counter()
    sample_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                # 嘗試Python字典格式
                data = ast.literal_eval(line)
            except:
                try:
                    # 嘗試JSON格式
                    data = json.loads(line)
                except:
                    continue
            
            # 收集鍵
            keys = list(data.keys())
            all_keys.update(keys)
            
            # 保存前幾個樣本
            if len(sample_data) < 10:
                sample_data.append(data)
    
    print(f"\n分析前 {sample_lines} 行...")
    print(f"找到 {len(all_keys)} 種不同的鍵:")
    print("-" * 40)
    
    for key, count in sorted(all_keys.items()):
        percentage = count / sample_lines * 100
        print(f"{key:20s}: {count:6d} 次 ({percentage:5.1f}%)")
    
    print(f"\n鍵出現總次數: {sum(all_keys.values())}")
    print(f"平均每行的鍵數: {sum(all_keys.values()) / sample_lines:.1f}")
    
    # 顯示樣本數據
    print(f"\n前3個樣本數據:")
    print("-" * 40)
    for i, data in enumerate(sample_data):
        print(f"\n樣本 {i+1}:")
        for key, value in data.items():
            value_type = type(value).__name__
            if isinstance(value, str):
                preview = value[:50] + "..." if len(value) > 50 else value
                print(f"  {key}: '{preview}' ({value_type})")
            elif isinstance(value, list):
                print(f"  {key}: 列表[{len(value)}] {value[:3]}{'...' if len(value) > 3 else ''}")
            elif isinstance(value, dict):
                print(f"  {key}: 字典[{len(value)}] 鍵: {list(value.keys())[:3]}{'...' if len(value) > 3 else ''}")
            else:
                print(f"  {key}: {value} ({value_type})")

if __name__ == "__main__":
    # 修改為你的文件路徑
    META_FILE = './raw_amazon_insturments/meta_Musical_Instruments.json'
    inspect_metadata_keys(META_FILE, sample_lines=1000)