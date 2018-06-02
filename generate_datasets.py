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
import random
import traceback
from PIL import Image


CSV_FILE = 'CSV-files/train.csv'
OUTDIR = 'TrainDatasets/'
IMAGES = 'Images/'
DATASET_DIR = 'Train' # Dont add the '/'
NUM_DATASETS = 5
NUM_WORKERS = 8
NEW_H = 224
NEW_W = 224

class Loader():
    def __init__(self):
        self.lid_to_imgs = None
        self.datasets = None
        self.num_downsamples = 0
        self.found_full = False

    def load_lid_to_imgs(self):
        '''
        Load file to global lid_to_imgs from pickle file or creates it and makes a pickle.
        '''
        # Load or create lid_to_imgs
        lid_to_imgs_pickle = 'lid_to_imgs.pickle'
        if os.path.isfile(lid_to_imgs_pickle):
            with open(lid_to_imgs_pickle, 'rb') as p:
                self.lid_to_imgs = pickle.load(p)

        else:
            # Maps landmark_id to a list of id with that l_id
            self.lid_to_imgs = defaultdict(list)
            for row in tqdm.tqdm(train_data.iterrows(), total=train_data.shape[0]):
                l_id = row[1]['landmark_id']
                img_id = row[1]['id'] # image
                self.lid_to_imgs[l_id].append(img_id)
            with open(lid_to_imgs_pickle, 'wb') as p:
                pickle.dump(self.lid_to_imgs, p, protocol=pickle.HIGHEST_PROTOCOL)

    def get_labels(self, csv_file):
        '''
        Returns dict: labels_tuple: list of (index, l_id, l_id_count, label). Index unique id from 0 to 14950 (14951 total), where label is upsample_{1-5} or downsample
        '''
        train_data = pd.read_csv(csv_file)
        l_id_counts = train_data['landmark_id'].value_counts()

        labels_tuples = []
        for index, l_id_with_count_tuple in enumerate(l_id_counts.iteritems()):
            l_id, l_id_count = l_id_with_count_tuple
            if l_id_count == 1:
                label = 'upsample_0'
            elif l_id_count < 5:
                label = 'upsample_1'
            elif l_id_count < 15:
                label = 'upsample_2'
            elif l_id_count < 30:
                label = 'upsample_3'
            elif l_id_count < 60:
                label = 'upsample_4'
            else:
                label = 'downsample'
            labels_tuples.append((index, l_id, l_id_count, label))
        return labels_tuples


    def transformImages(self, l_id, id_to_transformations):
        '''
        Transforms images and adds them to corresponding directories.
        '''
        for i in range(NUM_DATASETS):
            labels_dir = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id))
            if not os.path.exists(labels_dir):
                os.mkdir(labels_dir)

        for img_id in self.lid_to_imgs[l_id]:
            img_file = os.path.join(IMAGES, '{}.jpg'.format(img_id))
            if os.path.isfile(img_file):

                with Image.open(img_file) as img:
                    for transformations in id_to_transformations[img_id]: # id_to_transformations[img_id] is a list like ['c', 'cd']
                        new_img = img
                        if self.found_full == False and len(transformations) == 5:
                            print('label with all transformations: {}', l_id)
                            self.found_full = True

                        for t in transformations: # transformations is a string
                            if t == 'd':
                                new_img = new_img.point(lambda p: p * 0.8)
                            elif t == 'B':
                                new_img = new_img.point(lambda p: p * 1.4)
                            elif t == 'g':
                                new_img = new_img.convert('1')
                            elif t == 'c':
                                x = random.randint(0, w-NEW_W-1)
                                y = random.randint(0, h-NEW_H-1)
                                new_img = new_img.crop((x,y, x+NEW_W, y+NEW_H))
                            elif t == 'f':
                                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                        # TODO: resize to NEW_H, NEW_W
                        new_img = new_img.resize((NEW_W, NEW_H))
                        for i in range(NUM_DATASETS):
                            new_file_name = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}_{}.jpg'.format(img_id, transformations))
                            new_img.save(new_file_name, format='JPEG')

            else:
                #print('Image {} does not exist.'.format(img_id))
                return



    def process_label(self, img_label):
        '''
        Proccesses label. Get images that correspond to that labels and figures out all 60 images (some new that are
        transformed) that will represent that label.
        '''
        def superset(list_):
            sets = [set([])]
            for n in list_:
                sets.extend([s | {n} for s in sets])
            return sets

        try:

            # For upsampling: Get all images corresponding to that label (less than 60), and perform the relevant image transformations to each of them, and then sample 60 images from that.
            '''
            ImageTransformations:
            d: dim
            B: brighten
            g: grayscale
            c: crop
            f: flip
            '''

            (index, l_id, lid_count, label) = img_label
            # For downsampling: for every dataset, sample 60 random images that have the label l_id
            if label == 'downsample':
                for i in range(NUM_DATASETS):
                    downsamples = np.random.choice(self.lid_to_imgs[l_id], 60, replace=False)
                    self.datasets[i][index] = downsamples
                    self.num_downsamples += 1/NUM_DATASETS
                    # TODO Add sampled images into corresponding dataset

            elif label == 'upsample_0':   # flip (x2), dim/bright (x3), color-transform (x2), 4-crops (x5), total: x60
                # TODO implement all transformations
                pass
            else:
                lower_q = lid_count - 60 % lid_count
                upper_q = lid_count - lower_q
                lower = int(60 / lid_count)
                upper = lower + 1
                if label == 'upsample_1':   # flip (x2), dim (x2), Bright (double strength of dim)(x2), color-transform (x2), crop (x2), total: x32, downsample to 60
                    ss = superset(['f','d','B','g','c'])
                if label == 'upsample_2':   # flip (x2), dim (x2), Bright (x2), crop (x2), total: x16, downsample to 60
                    ss = superset(['f','d','B','c'])
                if label == 'upsample_3':   # flip (x2), crop (x2), total: x4, downsample to 60
                    ss = superset(['f','c'])
                if label == 'upsample_4':
                    ss = superset(['f'])
                id_to_transformations = { img: [''.join(s) for s in ss] for img in self.lid_to_imgs[l_id] }  # Maps img_id to a list of transformations that must be done to that image.
                lower_sample = random.sample(list(id_to_transformations), lower_q)  # lower_sample: image_ids that will use lower_q tranformations
                upper_sample = set(list(id_to_transformations)) - set(lower_sample) # upper_sample: image_ids that will use upper_q tranformation
                for img in lower_sample:
                    id_to_transformations[img] = random.sample(id_to_transformations[img], lower)
                for img in upper_sample:
                    id_to_transformations[img] = random.sample(id_to_transformations[img], upper)

                # Make sure we produce the correct number of upsamples.
                all_transformations = []
                for img in id_to_transformations:
                    for t in id_to_transformations[img]:
                        all_transformations.append(t)
                assert(len(all_transformations) == 60)

                self.transformImages(l_id, id_to_transformations)


            return 0
        except Exception as e:
            print('Unable to process l_id: {}, label: {}. Error: {}'.format(l_id, label, e))
            print(traceback.format_exc())
            return 1


    def process_labels(self, labels):
        for img_label in tqdm.tqdm(labels):
            self.process_label(img_label)
        print('num_downsamples: ', self.num_downsamples)


    def run(self):

        # Make relevant directories
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        for i in range(NUM_DATASETS):
            dataset_dir = os.path.join(OUTDIR, DATASET_DIR + str(i))
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)

        # Load pickle file: lid_to_imgs (maps { l_lid to list of images with that label }
        self.load_lid_to_imgs()
        labels = self.get_labels(CSV_FILE)  # Produce labels. maps { l_id: label_of(l_id) }
        total_labels = len(labels)
        print('num labels: {}'.format(total_labels))


        self.datasets = [[[]]*total_labels]*NUM_DATASETS  # TODO: take out, just for testing

        # Load
        self.process_labels(labels)


        for i in range(NUM_DATASETS):
            dataset = list(itertools.chain.from_iterable(self.datasets[i]))
            print('Num elements in dataset: {} '.format(len(dataset)))

def removeEmptyFolders(path):
  'Function to remove empty folders'
  if not os.path.isdir(path):
    return

  # remove empty subfolders
  files = os.listdir(path)
  if len(files):
    for f in files:
      fullpath = os.path.join(path, f)
      if os.path.isdir(fullpath):
        removeEmptyFolders(fullpath)

  # if folder empty, delete it
  files = os.listdir(path)
  if len(files) == 0:
    os.rmdir(path)

def main():
    loader = Loader()
    loader.run()
    removeEmptyFolders(OUTDIR)

    # TODO: try to add multiprocessing for transformImages. Make this a separate function after we load all needed data structures. YES we should be able to multiprocess this.
    # Simply get lid_to_imgs from the Generator.


if __name__ == '__main__':
    main()
