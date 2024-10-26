import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torch.optim as optim
from torch.optim import lr_scheduler
import mediapipe as mp
import cv2
import numpy as np

import matplotlib.pyplot as plt

def get_joints(image_path):
    # Get image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use mediapipe to get hand joint positions
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    results = hands.process(image)

    # Pre define landmarks in case there are missing hands or joints
    all_landmarks = [[(0,0)] * 21, [(0,0)] * 21]

    # Finding if mediapipe found right or left hand first
    rightFirst = False
    try:
        if results.multi_handedness[0].classification[0].index == 0 and results.multi_handedness[0].classification[0].label == "Right":
            rightFirst = True
    except:
        pass
    
    # Verifying dataset as mediapipe may not see every joint due to lighting issues or something else
    try:
        if not (results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2):
            print(image_path, "doesn't have 42 joints or 2 hands, training anyways")
            if len(results.multi_hand_landmarks[0].landmark) != 21:
                print("missing 21 joints for hand 0")
            if len(results.multi_hand_landmarks[1].landmark) != 21:
                print("missing 21 joints for hand 1")
    except:
        pass
    
    # Find joint positions in both hands
    try:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmark = [(0,0)] * 21
            for joint_index, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                hand_landmark[joint_index] = (x, y)
            if hand_index == 0:
                all_landmarks[rightFirst] = hand_landmark
            else:
                all_landmarks[not rightFirst] = hand_landmark
    except Exception:
        print("failed to get joint positions:" + image_path)
    
    return all_landmarks

class FingerSpellingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerSpellingModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # Bidirectional LSTM, so *2
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        print(x)
        concatenated_samples = [torch.cat(sample, dim=0) for sample in x[0]]

        # Stack samples to form a batch
        batch = torch.stack(concatenated_samples)

        # RNN layer
        out, _ = self.rnn(batch)

        # Fully connected layers
        out = F.relu(self.fc1(out[:, -1, :]))  # Get the last time step's output
        out = self.fc2(out)

        return out

class FingerSpellingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = self._load_data()
    
    def _load_data(self):
        data = []
        hmm = False
        for letter in os.listdir(self.root_dir):
            print("letter:", letter)
            letter_path = os.path.join(self.root_dir, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                joints = get_joints(image_path)
                label = int(letter)
                data.append((joints, label))
            if hmm:
                break
            else:
                hmm = True
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        joints, label = self.data[idx]
        return joints, label

def create_data_loader(root_dir, batch_size=32):
    dataset = FingerSpellingDataset(root_dir)

    print("Splitting dataset")
    train_size = int(0.7 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print("Creating training loader")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Creating validation loader")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader

def train_model(model, train_loader, validation_loader, epochs=200, lr=0.0001, weight_decay=0.0001):
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
    tolerance = 5
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
            for inputs, labels in validation_loader:
                outputs = model(input)

                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                print("predicted:", predicted, "actual:", labels)
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
        scheduler.step()
        
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
    # Variables
    root_dir = "./dataset"
    input_size = 84
    hidden_size = 10
    num_classes = 24

    # Define model
    model = FingerSpellingModel(input_size, hidden_size, num_classes)

    # Create the dataset as a data loader
    print("Creating data loaders")
    train_loader, validation_loader = create_data_loader(root_dir)

    # Train the model using data loader
    print("Training model")
    train_model(model, train_loader, validation_loader)

if __name__ == "__main__":
    main()

