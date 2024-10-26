import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import mediapipe as mp
import cv2
import numpy as np
import os
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

class FingerspellingModel(nn.Module):
    def __init__(self):
        super(FingerspellingModel, self).__init__()
        self.fc1 = nn.Linear(84, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 24)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.softmax(self.fc7(x), dim=1)
        return x

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

def create_data_loader(root_dir):
    # Get joint data
    train_data = []
    validation_data = []
    for letter in os.listdir(root_dir):
        index = 1
        letter_dir = os.path.join(root_dir, letter)
        for filename in os.listdir(letter_dir):
            image_path = os.path.join(letter_dir, filename)
            joints = get_joints(image_path)
            if index % 10 < 7:
                train_data.append((joints, letter))
            else:
                validation_data.append((joints, letter))
            index += 1

    # Creating data loaders from the dataset
    train_inputs = [torch.tensor(d[0], dtype=torch.float32) for d in train_data]
    train_labels = [int(d[1]) for d in train_data]
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_data_loader = torch.utils.data.TensorDataset(torch.stack(train_inputs), train_labels)
    train_data_loader = torch.utils.data.DataLoader(train_data_loader, shuffle=True)

    validation_inputs = [torch.tensor(d[0], dtype=torch.float32) for d in validation_data]
    validation_labels = [int(d[1]) for d in validation_data]
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    validation_data_loader = torch.utils.data.TensorDataset(torch.stack(validation_inputs), validation_labels)
    validation_data_loader = torch.utils.data.DataLoader(validation_data_loader)

    return train_data_loader, validation_data_loader

def train_model(model, train_data_loader, validation_data_loader, epochs=200, lr=0.000001, weight_decay=0.0001):
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
        for inputs, labels in train_data_loader:
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
            for inputs, labels in validation_data_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

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
    # Directory of dataset
    root_dir = "./dataset"

    # Define model
    model = FingerspellingModel()

    # Create the dataset as a data loader
    print("Creating data loaders")
    train_data_loader, validation_data_loader = create_data_loader(root_dir)

    # Train the model using data loader
    print("Training model")
    train_model(model, train_data_loader, validation_data_loader)

if __name__ == "__main__":
    main()