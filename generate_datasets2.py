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
import threading


CSV_FILE = 'CSV-files/train.csv'
OUTDIR = 'ReducedTrainDatasets/'
VAL_DIR = 'ReducedValidationDataset/'
IMAGES = 'Images/'
DATASET_DIR = 'Train' # Dont add the '/'
VAL_PER_LID = 2
NUM_DATASETS = 3
NUM_WORKERS = 8
NEW_H = 224
NEW_W = 224

class Loader():
    '''
    Generates NUM_DATASETS from ImagesInDir and generates a validation set with images that will NOT be included by any means on the training set.
    Organized by class folder, with 60-120 images per class.
    '''
    def __init__(self):
        self.id_to_lid = None     # Maps { img_id: l_id }
        self.lid_to_imgs = None   # Maps { l_id: [ img_ids with l_id ]}
        self.total_labels = None  # Total number of labels in downloaded images.
        self.num_classes_ignored = 0

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

            # without pandas, so we only use images that were downloaded
            for k_v in tqdm.tqdm(self.id_to_lid.items()):
                img_id, l_id = k_v
                self.lid_to_imgs[l_id].append(img_id)

            with open(lid_to_imgs_pickle, 'wb') as p:
                pickle.dump(self.lid_to_imgs, p, protocol=pickle.HIGHEST_PROTOCOL)

    def get_labels(self, csv_file):
        '''
        Returns dict: labels_tuple: list of (index, l_id, l_id_count, label). Index unique id from 0 to 14950 (14951 total), where label is upsample_{1-5} or downsample
        '''

        downloaded_imgs = set(os.listdir(IMAGES))

        # Load id to lid dict.
        self.id_to_lid = {}
        with open(CSV_FILE) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_id = row['id']
                img_file = '{}.jpg'.format(img_id)
                if img_file in downloaded_imgs:
                    self.id_to_lid[img_id] = row['landmark_id']

        # Load lid_counts dict.
        l_id_counts = defaultdict(int)
        for img_file in downloaded_imgs:
            img_id = img_file[:-4]
            l_id = self.id_to_lid[img_id]
            l_id_counts[l_id] += 1

        labels_tuples = []
        for index, l_id_with_count_tuple in enumerate(l_id_counts.items()):
            l_id, l_id_count = l_id_with_count_tuple
            label = None
            if l_id_count < 10:
                label = 'ignore'
            elif l_id_count < 15 + VAL_PER_LID:
                label = 'upsample_1'
            elif l_id_count < 30 + VAL_PER_LID:
                label = 'upsample_2'
            elif l_id_count < 60 + VAL_PER_LID:
                label = 'upsample_3'
            elif l_id_count < 120 + VAL_PER_LID:
                label = 'move'
            else:
                label = 'downsample'
            labels_tuples.append((index, l_id, l_id_count, label))
        return labels_tuples


    def transformImages(self, l_id, id_to_transformations, label_images):
        '''
        Transforms images and adds them to corresponding directories.
        '''
        for img_id in label_images:
            img_file = os.path.join(IMAGES, '{}.jpg'.format(img_id))
            with Image.open(img_file) as img:
                for transformations in id_to_transformations[img_id]: # id_to_transformations[img_id] is a list like ['c', 'cd']
                    new_img = img

                    for t in transformations: # transformations is a string
                        if t == 'd':
                            new_img = new_img.point(lambda p: p * 0.8)
                        elif t == 'B':
                            new_img = new_img.point(lambda p: p * 1.4)
                        elif t == 'g':
                            new_img = new_img.convert('L')
                        elif t == 'c':
                            w, h = img.size
                            CROP_W = 200
                            CROP_H = 200
                            x = random.randint(0, w-CROP_W-1)
                            y = random.randint(0, h-CROP_H-1)
                            new_img = new_img.crop((x,y, x+CROP_W, y+CROP_H))
                        elif t == 'f':
                            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_img = new_img.resize((NEW_W, NEW_H))
                    for i in range(NUM_DATASETS):
                        new_file_name = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}_{}.jpg'.format(img_id, transformations))
                        new_img.save(new_file_name, format='JPEG')

        for i in range(NUM_DATASETS):
            new_imgs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id)))
            assert(len(new_imgs) == 60)


    def process_label(self, img_label):
        '''
        Proccesses label. Get images that correspond to that labels and figures out all 60 images (some new that are
        transformed) that will represent that label
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
            B: brighten 1.4
            b: brighten 1.3
            g: grayscale
            c: crop
            f: flip
            R: resized
            '''

            (index, l_id, lid_count, label) = img_label

            if label == 'ignore':
                return 

            lid_count -= VAL_PER_LID

            # Create directories for the label in each dataset.
            for i in range(NUM_DATASETS):
                label_dir = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id))
                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)

            # Add dir to validation set.
            if not os.path.exists(VAL_DIR):
                os.mkdir(VAL_DIR)
            label_dir_val = os.path.join(VAL_DIR, str(l_id))
            if not os.path.exists(label_dir_val):
                os.mkdir(label_dir_val)

            # Get validation imgs
            label_images = self.lid_to_imgs[l_id].copy()
            to_val = np.random.choice(label_images, VAL_PER_LID, replace=False)
            for img in to_val:
                label_images.remove(img)
            for img_id in to_val:
                img_file_path = os.path.join(IMAGES, '{}.jpg'.format(img_id))
                with Image.open(img_file_path) as img:
                    new_img_file_path = os.path.join(VAL_DIR, str(l_id), '{}.jpg'.format(img_id))
                    img.resize((NEW_H,NEW_W)).save(new_img_file_path, format='JPEG')

            assert(len(os.listdir(os.path.join(VAL_DIR, str(l_id)))) == VAL_PER_LID)

            # For downsampling: for every dataset, sample 120 random images that have the label l_id
            # Move if label is nothing
            DOWNSAMPLE_TO = 120
            if label == 'downsample' or label == 'move':
                for i in range(NUM_DATASETS):
                    if label == 'downsample':
                        label_images = np.random.choice(label_images, DOWNSAMPLE_TO, replace=False)
                    for img_id in label_images:
                        img_file_path = os.path.join(IMAGES, '{}.jpg'.format(img_id))
                        with Image.open(img_file_path) as img:
                            new_img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}.jpg'.format(img_id))
                            img.resize((NEW_H,NEW_W)).save(new_img_file_path, format='JPEG')
                    new_imgs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id)))
                    if label == 'move':
                        assert(len(new_imgs) == lid_count)
                    elif label == 'downsample':
                        assert(len(new_imgs) == DOWNSAMPLE_TO)

            else:
                lower_q = lid_count - 60 % lid_count
                upper_q = lid_count - lower_q
                lower = int(60 / lid_count)
                upper = lower + 1
                if label == 'upsample_1':   # flip (x2), dim (x2), Bright (x2), crop (x2), total: x16, downsample to 60
                    ss = superset(['f','d','B','c'])
                if label == 'upsample_2':   # flip (x2), crop (x2), total: x4, downsample to 60
                    ss = superset(['f','c'])
                if label == 'upsample_3':
                    ss = superset(['f'])
                id_to_transformations = { img: [''.join(s) for s in ss] for img in label_images }  # Maps img_id to a list of transformations that must be done to that image.
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

                self.transformImages(l_id, id_to_transformations, label_images)


            return 0
        except Exception as e:
            print('Unable to process l_id: {}, label: {}. Error: {}'.format(l_id, label, e))
            print(traceback.format_exc())
            return 1


    def process_labels_threaded(self, labels):

        def thread_target(t_id):
            global c
            global num_done
            while True:
                c.acquire()
                remaining = len(labels)
                if remaining % 500 == 0:
                    print('Labels remaining: {}'.format(remaining))
                if remaining == 0:
                    num_done += 1
                    if num_done == NUM_WORKERS:
                        c.notify_all()
                    c.release()
                    return

                img_label = labels.pop()
                (_, _,_, label) = img_label
                if label == 'ignore':
                    self.num_classes_ignored += 1
                c.release()
                self.process_label(img_label)

        global c
        c = threading.Condition()
        c.acquire()

        # Start and join threads
        threads = [ threading.Thread(target=thread_target, args=(str(i))) for i in range(NUM_WORKERS) ]

        global num_done
        num_done = 0
        for t_id, t in enumerate(threads):
            t.start()
        c.wait()
        c.release()
        for t_id, t in enumerate(threads):
            threads[t_id].join()

        print('DONE')


    def run(self):

        # Make Train directories
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        for i in range(NUM_DATASETS):
            dataset_dir = os.path.join(OUTDIR, DATASET_DIR + str(i))
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)


        labels = self.get_labels(CSV_FILE)  # Produce labels. maps { l_id: label_of(l_id) }
        self.total_labels = len(labels)
        print('num labels: {}'.format(self.total_labels))

        # Load pickle file: lid_to_imgs (maps { l_lid to list of images with that label }
        self.load_lid_to_imgs()

        # Load
        self.process_labels_threaded(labels)

        print('Checking that we have the correct number of class directories: total_labels {} - num_ignored {} = {}.'.format(self.total_labels, self.num_classes_ignored, self.total_labels - self.num_classes_ignored))
        class_dirs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(0)))
        if len(class_dirs) != self.total_labels - self.num_classes_ignored:
            print('[ERROR] There are {} class directories in train set 0.'.format(len(class_dirs)))
        else:
            print('Check passed!')
        class_dirs_val = os.listdir(VAL_DIR)
        if len(class_dirs_val) != self.total_labels - self.num_classes_ignored:
            print('[ERROR] There are {} class directories in the validation set.'.format(len(class_dirs_val)))
        else:
            print('Check passed!')

        print('Making sure all directories have between 60 and 120 images.')
        for i in tqdm.tqdm(range(NUM_DATASETS)):
            for class_dir in class_dirs:
                class_dir_path = os.path.join(OUTDIR, DATASET_DIR + str(i), class_dir)
                if len(os.listdir(class_dir_path)) < 60 or len(os.listdir(class_dir_path)) > 120 :
                    print('[ERROR] Dir {} does not have between 60 and 120 images in it!!'.format(class_dir_path))
        print('DONE!')

def main():
    loader = Loader()
    loader.run()



if __name__ == '__main__':
    main()
