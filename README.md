# Codebase For Our FYP
Created an iOS app which uses the model to recognise Auslan fingerspellings.
![iOS Application](/readme/ios-application.png)

Trained a model with ASL and Auslan dataset.
![Sohn's Model Architecture](/readme/Sohn's%20Model%20Architrecture.png)
Source: https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434588

# File Structure
1. fyp-initial-experiments
This folder contains our early experiments on testing mediapipe and creating several model architectures and attempting to train them.

2. fyp-inference-tests
This folder contains the files relating to investigating what the parquet files are and getting Sohn's model working with live camera input.

3. /fyp-train
This folder contains the files relating to processing the video data, converting it to parquets and tfrecords and training it on Sohn's model.

4. /ios-application
This folder contains the files relating to the ios application.

# Sources
https://www.kaggle.com/competitions/asl-fingerspelling<br>
https://www.kaggle.com/code/hoyso48/2nd-place-solution-training<br>
https://www.kaggle.com/code/hoyso48/aslfr-create-tfr