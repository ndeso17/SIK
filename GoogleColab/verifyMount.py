import os

DATASET_DIR = "/content/drive/MyDrive/KFR/Kuliah/PCD/SPARX/Record/Training"

print("Isi folder SPARXTrain:")
for item in os.listdir(DATASET_DIR):
    print(" -", item)

print("\nIsi folder images:")
for img in os.listdir(os.path.join(DATASET_DIR, "anotasi_img"))[:10]:
    print(" -", img)
