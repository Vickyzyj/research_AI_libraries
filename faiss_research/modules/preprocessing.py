import os
from PIL import Image

# Create image dictionary
def create_img_dict(folder_path):
    img_dict = {}
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue

        animal_type = os.path.basename(root)
        img_dict[animal_type] = [f for f in files]

    return img_dict

# Load images and labels to lists
def create_img_list(img_dict, folder_path):
    img_names = {i: list(img_dict.keys())[i] for i in range(len(img_dict.keys()))}
    img_list = []

    for i, name in enumerate(img_dict.keys()):
        for img_file in img_dict[name]:
            with Image.open(os.path.join(folder_path, name, img_file)) as img:
                img_list.append((img.convert('RGB'), i))

    labels = [_label for _img, _label in img_list]

    return img_list, labels, img_names
