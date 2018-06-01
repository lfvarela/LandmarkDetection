import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import defaultdict
import tqdm
import csv
import multiprocessing
import pickle
import itertools

CSV_FILE = 'CSV-files/train.csv'
OUTDIR = 'TrainDatasets/'
DATASET_DIR = 'Train' # Dont add the '/'
NUM_DATASETS = 5
NUM_WORKERS = 8

class Generator():
    def __init__(self):
        self.lid_to_img = None
        self.test_dirs = None
        self.num_downsamples = 0

    def get_labels(self, csv_file):
        '''
        Returns dict: labels, maps { l_id: label_of(l_id) }, where labels are upsample_{1-5} or downsample
        Also load file to global lid_to_img from pickle or creates it and makes a pickle.
        '''
        train_data = pd.read_csv(csv_file)
        counts = train_data['landmark_id'].value_counts()

        # Load or create lid_to_img
        lid_to_img_pickle = 'lid_to_img.pickle'
        if os.path.isfile(lid_to_img_pickle):
            with open(lid_to_img_pickle, 'rb') as p:
                self.lid_to_img = pickle.load(p)

        else:
            # Maps landmark_id to a list of id with that l_id
            self.lid_to_img = defaultdict(list)
            for row in tqdm.tqdm(train_data.iterrows(), total=train_data.shape[0]):
                l_id = row[1]['landmark_id']
                img_id = row[1]['id'] # image
                self.lid_to_img[l_id].append(img_id)
            with open(lid_to_img_pickle, 'wb') as p:
                pickle.dump(self.lid_to_img, p, protocol=pickle.HIGHEST_PROTOCOL)

        labels = {} # Maps l_id to label for that l_id
        for l_id, l_id_count in counts.iteritems():
            if l_id_count < 5:
                label = 'upsample_1' # flip (x2), dim/bright (x3), color-transform (x2), crop (x5), total: x60, downsample to 60
            elif l_id_count < 15:
                label = 'upsample_2' # flip (x2), dim/bright (x3), crop (x2), total: x12, downsample to 60
            elif l_id_count < 30:
                label = 'upsample_3' # flip (x2), crop (x2), total: x4, downsample to 60
            elif l_id_count < 60:
                label = 'upsample_4' # rotate (x2), downsample to 60
            else:
                label = 'downsample' # take 60 random images for each directory
            labels[l_id] = label

        labels_tuples = []
        for index, k in enumerate(labels):
            labels_tuples.append((index, k, labels[k]))
        return labels_tuples



    def process_label(self, img_label):
        '''
        For now adds do
        '''
        try:
            (index, l_id, label) = img_label
            for i in range(NUM_DATASETS):

                # Add downsamples to the relevant directory and index
                if label == 'downsample':
                    downsamples = np.random.choice(self.lid_to_img[l_id], 60, replace=False)
                    self.test_dirs[i][index] = downsamples
                    self.num_downsamples += 1/NUM_DATASETS
            return 0
        except Exception as e:
            print('Unable to process l_id: {}, label: {}. Error: {}'.format(l_id, label, e))
            return 1

    def process_labels(self, labels):
        for img_label in tqdm.tqdm(labels):
            self.process_label(img_label)
        print('num_downsamples: ', self.num_downsamples)


    def run(self):
        labels = self.get_labels(CSV_FILE)  # maps { l_id: label_of(l_id) }

        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

        for i in range(NUM_DATASETS):
            dataset_dir = os.path.join(OUTDIR, DATASET_DIR + str(i))
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)

        total_labels = len(labels)
        print('num labels: {}'.format(total_labels))
        self.test_dirs = [[[]]*total_labels]*NUM_DATASETS  # TODO: take out, just for testing
        self.process_labels(labels)


        for i in range(NUM_DATASETS):
            dataset = list(itertools.chain.from_iterable(self.test_dirs[i]))
            print('Num elements in dataset: {} '.format(len(dataset)))

def main():
    g = Generator()
    g.run()


if __name__ == '__main__':
    main()
