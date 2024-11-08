import os
import sys

def check_dll_paths():
    # 檢查 PATH 環境變量
    path = os.environ.get('PATH', '').split(os.pathsep)
    
    # 常見的 Intel oneAPI 路徑
    intel_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl",
        r"C:\Program Files (x86)\Intel\oneAPI\tbb"
    ]
    
    print("檢查系統路徑:")
    for p in path:
        if 'Intel' in p or 'oneAPI' in p:
            print(f"找到 Intel 路徑: {p}")
            
    print("\n檢查常見 Intel 路徑是否存在:")
    for p in intel_paths:
        if os.path.exists(p):
            print(f"存在: {p}")
        else:
            print(f"不存在: {p}")

if __name__ == "__main__":
    check_dll_paths() 