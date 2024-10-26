import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class FingerSpellingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = self.load_images()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_images(self):
        images = []
        for letter in os.listdir(self.root_dir):
            letter_dir = os.path.join(self.root_dir, letter)
            for filename in os.listdir(letter_dir):
                images.append((os.path.join(letter_dir, filename), letter))
        
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        label = int(label)

        return image, label
    
class FingerspellingModel(nn.Module):
    def __init__(self):
        super(FingerspellingModel, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2 * 21 * 3, 1024, bias=False)
        self.dropout1 = nn.Dropout(0.275)
        self.conformer = nn.Transformer(d_model=1024, nhead=8, num_encoder_layers=1, dim_feedforward=4096, dropout=0.375)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        self.dropout2 = nn.Dropout(0.275)
        self.fc3 = nn.Linear(256, 24)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout1(x)

        x = x.unsqueeze(1)  # Add sequence dimension for the transformer
        x = self.conformer(x, x)
        x = x.squeeze(1)  # Remove sequence dimension

        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, epochs=200, lr=0.00000005, weight_decay=0.0001):
    # Use cross entropy loss for identifying single class
    criterion = nn.CrossEntropyLoss()

    # Machine learning algorithm
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler to decrease learning rate
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Variables for graphing
    train_accs = []
    valid_accs = []

    # Variables for early stopping
    tolerance = 30
    counter = 0
    prev_acc = 0
    min_delta = 0.01

    for epoch in range(epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                print('predicted:', predicted, 'actual:', labels)
        valid_acc = correct_valid / total_valid

        # Store accuracies for plotting
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        # Print results for this epoch
        print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc}, Valid Acc: {valid_acc}")

        # Check for early stopping
        if valid_acc - prev_acc < min_delta:
            counter += 1
        else:
            counter = 0
        if counter >= tolerance:
            break

        prev_acc = valid_acc

        # Step scheduler
        #scheduler.step()
        
    # Plotting accuracy graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(valid_accs)+1), valid_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.show()

    # Saving the trained model
    torch.save(model.state_dict(), 'fingerspelling_model.pth')

def main():
    # Create datasets
    root_dir = "./image_representation"
    dataset = FingerSpellingDataset(root_dir)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define model
    model = FingerspellingModel()

    # Create dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()