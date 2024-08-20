import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# Load FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define the path to your dataset directory
dataset_path = 'dataset'

# Define a function to generate embeddings from images
def generateEmbeddings(imagePaths):
    embeddings = []
    labels = []

    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('RGB')
        img = transforms.Resize((160, 160))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img).cpu().numpy().flatten()

        embeddings.append(embedding)
        
        # Extract user ID from the image file name
        user_id = int(os.path.basename(imagePath).split(".")[1])
        labels.append(user_id)

    return np.array(embeddings), np.array(labels)

# Train the classifier
def trainClassifier(embeddings, labels):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear')
    classifier.fit(embeddings, labels)
    return classifier

# Get embeddings and labels
def getEmbeddingsAndLabels(dataset_path):
    imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    return generateEmbeddings(imagePaths)

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
embeddings, labels = getEmbeddingsAndLabels(dataset_path)

# Train classifier
classifier = trainClassifier(embeddings, labels)

# Save the classifier
import joblib
joblib.dump(classifier, 'trainer/classifier.pkl')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(labels))))
