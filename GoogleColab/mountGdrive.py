import os
from google.colab import drive
base_path = "/content/drive"
drive.mount(base_path, force_remount=True)
print("Isi /content/drive:")
for item in os.listdir(base_path):
    print(" -", item)
print("\nIsi /content/drive/MyDrive:")
for item in os.listdir(os.path.join(base_path, "MyDrive")):
    print(" -", item)
