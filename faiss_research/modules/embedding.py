import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np

# Create vectors from images
class ImageEmbedding:
    def __init__(self):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(self.model.children()))[:-1]
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

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
    images_list = ['test_1.jpg', 'test_2.jpeg', 'test_3.jfif', 'test_4.jfif', 'test_5.jfif']
    images = []
    for img in images_list:
        _p = os.path.join(_path, img)
        images.append(Image.open(_p).convert('RGB'))

    print(len(images))

    img_embed = ImageEmbedding()
    embedded_list = img_embed.embed_list(images)
    print(embedded_list.shape)