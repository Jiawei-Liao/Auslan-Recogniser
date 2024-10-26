import os
import cv2
import mediapipe as mp

# Function to process image with Mediapipe and write output to text file
def process_image(image_path, output_folder):
    # Initialize Mediapipe hands module
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        # Read image
        image = cv2.imread(image_path)
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process image with Mediapipe
        results = hands.process(image_rgb)
        hand_landmarks = results.multi_hand_landmarks
        # Extract hand landmarks if available
        output_path = os.path.join(output_folder, f"{os.path.basename(image_path).replace('.png', '')}.txt")
        with open(output_path, 'w') as f:

            if hand_landmarks:
                for hand_index in range(0, 2):
                    f.write(f"{hand_index}\n")
                    if hand_index < len(hand_landmarks):
                        hand = hand_landmarks[hand_index]
                        for joint_index in range(0, 21):
                            if joint_index < len(hand.landmark):
                                landmark = hand.landmark[joint_index]
                                f.write(f"{landmark.x} {landmark.y} {landmark.z}\n")
                            else:
                                f.write("\n")

                    else:
                        for _ in range(0, 22):
                            f.write("\n")

# Function to create dataset folder for text outputs
def create_dataset(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate through image folders (a, b, c, etc.)
    for letter_folder in os.listdir(input_folder):
        letter_input_folder = os.path.join(input_folder, letter_folder)
        letter_output_folder = os.path.join(output_folder, letter_folder)
        # Create output folder for the letter if it doesn't exist
        if not os.path.exists(letter_output_folder):
            os.makedirs(letter_output_folder)
        # Process each image in the letter folder
        for image_file in os.listdir(letter_input_folder):
            image_path = os.path.join(letter_input_folder, image_file)
            # Process image and write output to text file
            print(image_path)
            print(letter_output_folder)
            process_image(image_path, letter_output_folder)

# Example usage
input_folder = 'dataset'  # Folder containing image folders (a, b, c, etc.)
output_folder = 'text_dataset'  # Output folder for text files
create_dataset(input_folder, output_folder)
