import kagglehub
import os
import shutil

path = kagglehub.dataset_download("karnikakapoor/art-portraits")
print("Downloaded to cache:", path)

TARGET_DIR = "data/art_portraits"
os.makedirs(TARGET_DIR, exist_ok=True)

for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(TARGET_DIR, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset ready at:", TARGET_DIR)
