import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

# 1. Define a simple CNN model for face recognition
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Batch normalization layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512) # fully connected layer
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. Create a custom dataset for face images
class FaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        This class creates a dataset from a directory of images.
        The images are expected to be in the following format:
        root_dir/
            ├── person1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            ├── person2/
            │   ├── image1.jpg
            │   ├── image2.jpg
            └── ...
        The class labels are assigned based on the subdirectory names.
        Each image is converted to RGB format and resized to 128x128 pixels.

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls_name in self.classes:
            class_path = os.path.join(root_dir, cls_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 3. Training function
def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 20) -> nn.Module:
    """
    Train the model using the provided data loader, loss function, and optimizer.
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to train the model on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
    Returns:
        model (nn.Module): The trained model.
    """
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

# 4. Real-time face recognition function
def recognize_faces_realtime(model: nn.Module, device: torch.device, class_names: list, confidence_threshold: float = 0.5) -> None:
    """
    Perform real-time face recognition using the webcam (using cv2).
    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run the model on (CPU or GPU).
        class_names (list): List of class names for the recognized faces.
        confidence_threshold (float): Minimum confidence to consider a prediction valid.
    """
    # Initialize MTCNN for face detection: MTCNN is used to detect faces in the image
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    model.eval()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        # Detect faces
        boxes, _ = mtcnn.detect(image)
        
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Extract face
                face = image.crop((x1, y1, x2, y2))
                
                # Preprocess face
                face_tensor = transform(face).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted = torch.max(probabilities, 1)
                    
                    if max_prob.item() > confidence_threshold:
                        person_name = class_names[predicted.item()]
                        label = f"{person_name}: {max_prob.item():.2f}"
                    else:
                        label = "Unknown"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to tie everything together
def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths
    data_dir = "faces_dataset"  # Folder with subfolders for each person
    model_path = "face_recognition_model.pth"
    
    # Create dataset directory structure if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created dataset directory at {data_dir}")
        print("Please organize your face images in the following structure:")
        print(f"{data_dir}/person1/image1.jpg")
        print(f"{data_dir}/person1/image2.jpg")
        print(f"{data_dir}/person2/image1.jpg")
        print("... and so on")
        return
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = FaceDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Check if we have data
    if len(dataset) == 0:
        print("No images found in the dataset directory!")
        return
    
    num_classes = len(dataset.classes)
    print(f"Found {num_classes} classes: {dataset.classes}")
    
    # Create model
    model = FaceRecognitionModel(num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Check if model exists, otherwise train
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
    else:
        print("Training model...")
        model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=20)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Start real-time recognition
    print("Starting real-time face recognition. Press 'q' to quit.")
    recognize_faces_realtime(model, device, dataset.classes)

if __name__ == "__main__":
    main()