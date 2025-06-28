
from modules.preprocessing import create_img_dict, create_img_list
from modules.embedding import ImageEmbedding          # suitable for general image recognition
# from modules.embeddingV2 import ImageEmbedding      # suitable for face recognition
from modules.faiss_search import FaissSearch

# Load images from folder
folder_path = '../storage/animal_images/animals'
animal_dict = create_img_dict(folder_path)
animal_img_list, labels, animal_names = create_img_list(animal_dict, folder_path)
print('Number of animal images:', len(animal_img_list))

# Prepare training images (convert to vectors)
preprocessed_img_list = [img for img, name in animal_img_list]
img_embed = ImageEmbedding()
training_dataset = img_embed.embed_list(preprocessed_img_list)

# Initiate FAISS
animal_search = FaissSearch()

# 1. Flat Search
animal_search.flat_search(training_dataset, labels, animal_names)
save_dir = 'pickle_files/Flat_index.pkl'
animal_search.save(save_dir)
print('Flat index file saved successfully.')

# 2. LSH Search
animal_search.lsh_search(training_dataset, labels, animal_names)
save_dir = 'pickle_files/LSH_index.pkl'
animal_search.save(save_dir)
print('LSH index file saved successfully')

# 3. HNSW Search
animal_search.hnsw_search(training_dataset, labels, animal_names)
save_dir = 'pickle_files/HNSW_index.pkl'
animal_search.save(save_dir)
print('HNSW index file saved successfully.')
