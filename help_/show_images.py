import os
import cv2
import matplotlib.pyplot as plt

image_folder = r"D:\Project\Traffic_signs\data_mine\images\48"

files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
files = sorted(files)[:10]


images = []
for f in files:
    img = cv2.imread(os.path.join(image_folder, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    images.append(img)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.axis("off")

plt.tight_layout()
plt.show()
