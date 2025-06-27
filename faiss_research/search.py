from modules.embedding import ImageEmbedding
from modules.faiss_search import FaissSearch
from PIL import Image
import os

def test_images(index, test_folder, ground_truth):
    test_img_files = []
    for root, dirs, files in os.walk(test_folder):
        test_img_files += [*files]

    img_embed = ImageEmbedding()
    count = len(ground_truth)
    correct_count = 0
    k = 10
    i = 0
    for img_file in test_img_files:
        try:
            img = Image.open(os.path.join(test_folder, img_file)).convert('RGB')
        except:
            continue

        img_q = img_embed.embed_list([img])
        label, D, I = index.search(img_q, k)
        correct_count += 1 if label==ground_truth[i] else 0
        i+=1
        print('Test Image', i, ':', label)

    print(f'Accuracy: {correct_count/count*100:.2f}%')
    print('\n')


# Initiate Test
test_folder = '../storage/animal_images/query'
gt_file = os.path.join(test_folder, 'image_labels.txt')
with open(gt_file, 'r') as f:
    gt = [line.strip() for line in f]

print(gt)

animal_search = FaissSearch()

# Flat Search:
print("Faiss Flat Search:")
load_dir = 'pickle_files/Flat_index.pkl'

# Queries:
saved_index = animal_search.load(load_dir)
test_images(saved_index, test_folder, gt)


# LSH Search:
print("Faiss LSH Search:")
animal_search = FaissSearch()
load_dir = 'pickle_files/LSH_index.pkl'

# Queries:
saved_index = animal_search.load(load_dir)
test_images(saved_index, test_folder, gt)


# HNSW Search:
print("Faiss HNSW Search:")
animal_search = FaissSearch()
load_dir = 'pickle_files/HNSW_index.pkl'

# Queries:
saved_index = animal_search.load(load_dir)
test_images(saved_index, test_folder, gt)


