import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np

# Create vectors from images
# Suitable for face recognition
class ImageEmbedding:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='casia-webface', classify=False).eval()
        self.transform = transforms.Compose([transforms.Resize((160,160)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        return embedding

    def embed_list(self, img_list):
        embedding_list = [self.get_embedding(img) for img in img_list]
        embedding_matrix = np.vstack(embedding_list).astype('float32')
        return embedding_matrix


if __name__ == '__main__':
    import os
    from pathlib import Path
    from PIL import Image
    _path = Path(os.getcwd()).parent.parent
    _path = os.path.join(_path, 'storage/animal_images/query')
    images_list = ['test_01.jpg', 'test_02.jpeg', 'test_03.jfif', 'test_04.jfif', 'test_05.jfif']
    images = []
    for img in images_list:
        _p = os.path.join(_path, img)
        images.append(Image.open(_p).convert('RGB'))

    print(len(images))

    img_embed = ImageEmbedding()
    embedded_list = img_embed.embed_list(images)
    print(embedded_list.shape)