from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Load the pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load and allign face with MTCNN
mtcnn = MTCNN(image_size=160, margin=20)

# Load and preprocess images
img = Image.open('face4.jpg')
face_tensor = mtcnn(img) #outputs 3x160x160 tensor
img2 = Image.open('face5.jpg')
face2_tensor = mtcnn(img2)

with torch.no_grad():
    emb1 = model(face_tensor.unsqueeze(0))
    emb2 = model(face2_tensor.unsqueeze(0))

to_pil = T.ToPILImage()
face_pil = to_pil(face_tensor)
plt.imshow(face_pil)
plt.axis('off')
plt.show()

# Cosine similarity
sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
print("Cosine similarity:", sim)