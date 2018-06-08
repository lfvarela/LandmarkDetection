# LandmarkDetection

Repository for our Google Landmark Recognition Challenge (Kaggle Competition). We finished with a gAP of 0.011, which would put us in the 198th place if we had submitted on time. These are satisfatory results given that we completed this in a couple of weeks using low computing resources as compared to top submissions. Our final project report is in the file 231-final-project.pdf.

## Directories

We have cleaned up the directories for better structure, note that file constants might have changed, so some constants might need to be changed to recreate our process. 

**Scripts Directory**: 

Important files include:

download_images.py: Download images from Kaggle csv file. We used code from: https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar

generate_datasets.py: Does preprocessing for our "first approach"

generate_datasets2.py: Does preprocessing for our "final approach"

images_to_dirs.py: Restructures the dataset by class labels

**Notebooks Directory**: 

DataGenerator.py: class used to load data into our training model using multiprocessing and multiple GPUs. 

data_viz.ipynb: Used for visualizing the data, getting label counts, percentiles, etc... Used some code from https://www.kaggle.com/codename007/a-very-extensive-landmark-exploratory-analysis

ensemble.ipynb: Used for the ensemble of multiple models.

evaluate.ipynb: Used to evaluate a model using the test dataset. 

new_train.ipynb: Model for training used on our "final approach".

train.ipynb: Model for training used on our "first approach".

**Zip Files Directory**: Includes zip files from Kaggle. 

**.gitignore:**

CSV-Files/: Add files downloaded from zip files here, or store somewhere else locally.

Images/ :Downloaded Images from Kaggle csv file

TrainDatasets/ : Structured train datasets, "first approach"

ImagesInDirs/ : Images by class

TestDataset/ : Test data

ReducedTrainDatasets/ : Structured train datasets, "final approach"

ReducedValidationDataset/ : Structured validation dataset, "final approach"

models/ : Saved models 

preds/ : numpy prediction arrays

Submissions/ : submission files for kaggle competition
