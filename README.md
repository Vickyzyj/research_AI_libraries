# Research_AI_Libraries
Research and practice on various AI libraries

### Project I. FAISS on Animal Image Recognition
#### Module Files:
* Preprocessing: [preprocessing.py](faiss_research/modules/preprocessing.py) Defined functions
  to load training images and labels to lists.
* Embedding: [embedding.py](faiss_research/modules/embedding.py) Defined ImageEmbedding class to
  to convert images to vectors using ResNet18.
* Faiss_search: [faiss_search.py](faiss_research/modules/faiss_search.py) Defined FaissSearch class which
  enables Flat search, LSH search, and HNSW search of FAISS library.

#### Test Files:
* [faiss_init.py](faiss_research/faiss_init.py): Prepare training data and create FAISS search pickle files.
* [search.py](faiss_research/search.py): Load FAISS search pickle files and recognize animals on test images.
* [FAISS_for_Image_Recognition.ipynb](FAISS_for_Image_Recognition.ipynb): An implementation of FAISS for 
  animal images recognition using Kaggle dataset. Overall accuracy was around 91%.
