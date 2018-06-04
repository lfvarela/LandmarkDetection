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
# OUTDIR = 'TrainDatasets/'
IMAGES = 'Images/'
DATASET_DIR = 'Train' # Dont add the '/'
NUM_DATASETS = 5
NUM_WORKERS = 4
NEW_H = 224
NEW_W = 224

class Loader():
    '''
    Generates NUM_DATASETS from Images, organized by class folder, with 60 images per class.
    '''
    def __init__(self):
        self.id_to_lid = None     # Maps { img_id: l_id }
        self.lid_to_imgs = None   # Maps { l_id: [ img_ids with l_id ]}
        self.total_labels = None  # Total number of labels in downloaded images.

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

        self.id_to_lid = {}
        with open(CSV_FILE) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_id = row['id']
                img_file = '{}.jpg'.format(img_id)
                if img_file in downloaded_imgs:
                    self.id_to_lid[img_id] = row['landmark_id']

        l_id_counts = defaultdict(int)
        for img_file in downloaded_imgs:
            img_id = img_file[:-4]
            l_id = self.id_to_lid[img_id]
            l_id_counts[l_id] += 1

        labels_tuples = []
        for index, l_id_with_count_tuple in enumerate(l_id_counts.items()):
            l_id, l_id_count = l_id_with_count_tuple
            if l_id_count == 1:
                label = 'upsample_0'
            elif l_id_count < 4:
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
        for img_id in self.lid_to_imgs[l_id]:
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
                            x = random.randint(0, w-NEW_W-1)
                            y = random.randint(0, h-NEW_H-1)
                            new_img = new_img.crop((x,y, x+NEW_W, y+NEW_H))
                        elif t == 'f':
                            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_img = new_img.resize((NEW_W, NEW_H))
                    for i in range(NUM_DATASETS):
                        new_file_name = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}_{}.jpg'.format(img_id, transformations))
                        new_img.save(new_file_name, format='JPEG')

        for i in range(NUM_DATASETS):
            new_imgs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id)))
            assert(len(new_imgs) == 60)


    def upsample_1_to_60(self, l_id):
        '''
        Apply transformations to a single image to bring it up to 60 samples.
        flip (x2), dim/bright (x3), color-transform (x2), 4-crops (x5), total: x60
        Save the image in all five datasets.
        '''
        assert(len(self.lid_to_imgs[l_id]) == 1)

        # Remove files in dir
        for i in range(NUM_DATASETS):
            class_dir = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id)))
            for img_file in class_dir:
                os.remove(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), img_file))

        img_id = self.lid_to_imgs[l_id][0]
        img_file = os.path.join(IMAGES, '{}.jpg'.format(img_id))

        try:

            # Transpose
            with Image.open(img_file) as img:
                transpose = img.transpose(Image.FLIP_LEFT_RIGHT)
                for i in range(NUM_DATASETS):
                    new_file_name = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}_{}.jpg'.format(img_id, ''))
                    img.save(new_file_name, format='JPEG')
                    new_file_name_t = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}_{}.jpg'.format(img_id, 't'))
                    transpose.save(new_file_name_t, format='JPEG')

            # Dim/Bright
            class_dir = os.path.join(OUTDIR, DATASET_DIR + str(0), str(l_id))
            new_imgs = os.listdir(class_dir)
            for img_file in new_imgs:
                img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(0), str(l_id), img_file)
                with Image.open(img_file_path) as img:
                    dim = img.point(lambda p: p * 0.7)
                    brighten = img.point(lambda p: p * 1.3)
                    for i in range(NUM_DATASETS):
                        new_file_name_d = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}{}.jpg'.format(img_file[:-4], 'd'))
                        new_file_name_b = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}{}.jpg'.format(img_file[:-4], 'b'))
                        dim.save(new_file_name_d, format='JPEG')
                        brighten.save(new_file_name_b, format='JPEG')

            new_imgs = os.listdir(class_dir)
            for img_file in new_imgs:
                img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(0), str(l_id), img_file)
                with Image.open(img_file_path) as img:
                    grey = img.convert('L')
                    for i in range(NUM_DATASETS):
                        new_file_name_g = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}{}.jpg'.format(img_file[:-4], 'g'))
                        grey.save(new_file_name_g, format='JPEG')

            new_imgs = os.listdir(class_dir)
            for img_file in new_imgs:
                img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(0), str(l_id), img_file)
                with Image.open(img_file_path) as img:
                    w, h = img.size
                    for j in range(4):
                        x = random.randint(0, w-NEW_W-1)
                        y = random.randint(0, h-NEW_H-1)
                        crop = img.crop((x,y, x+NEW_W, y+NEW_H))
                        for i in range(NUM_DATASETS):
                            new_file_name_c = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}{}{}.jpg'.format(img_file[:-4], 'c', str(j)))
                            crop.save(new_file_name_c, format='JPEG')

            # Resize them
            new_imgs = os.listdir(class_dir)
            for img_file in new_imgs:
                img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(0), str(l_id), img_file)
                with Image.open(img_file_path) as img:
                    resize = img.resize((NEW_H,NEW_W))
                    for i in range(NUM_DATASETS):
                        new_file_name_R = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}{}.jpg'.format(img_file[:-4], 'R'))
                        to_remove_file_path = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), img_file)
                        os.remove(to_remove_file_path)
                        resize.save(new_file_name_R, format='JPEG')

            new_imgs = os.listdir(class_dir)
            assert(len(new_imgs) == 60)

        except Exception as e:
            print('Error with PIL')
            print(traceback.format_exc())

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

            # Create directories for all the labels in each dataset.
            for i in range(NUM_DATASETS):
                labels_dir = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id))
                if not os.path.exists(labels_dir):
                    os.mkdir(labels_dir)

            # For downsampling: for every dataset, sample 60 random images that have the label l_id
            if label == 'downsample':
                for i in range(NUM_DATASETS):
                    downsamples = np.random.choice(self.lid_to_imgs[l_id], 60, replace=False)
                    for img_id in downsamples:
                        img_file_path = os.path.join(IMAGES, '{}.jpg'.format(img_id))
                        with Image.open(img_file_path) as img:
                            new_img_file_path = os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id), '{}.jpg'.format(img_id))
                            img.resize((NEW_H,NEW_W)).save(new_img_file_path, format='JPEG')
                    new_imgs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i), str(l_id)))
                    assert(len(new_imgs) == 60)

            elif label == 'upsample_0':   # flip (x2), dim/bright (x3), color-transform (x2), 4-crops (x5), total: x60
                self.upsample_1_to_60(l_id)

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

        print('Checking that we have {} number of class directories.'.format(self.total_labels))
        class_dirs = os.listdir(os.path.join(OUTDIR, DATASET_DIR + str(i)))
        if len(class_dirs) != self.total_labels:
            print('[ERROR] There are {} class directories.'.format(len(class_dirs)))
        else:
            print('Check passed!')

        print('Making sure all directories have 60 images.')
        for i in tqdm.tqdm(range(NUM_DATASETS)):
            for class_dir in class_dirs:
                class_dir_path = os.path.join(OUTDIR, DATASET_DIR + str(i), class_dir)
                if len(os.listdir(class_dir_path)) != 60:
                    print('[ERROR] Dir {} does not have 60 images in it!!'.format(class_dir_path))
        print('DONE!')

def main():
    loader = Loader()
    loader.run()



if __name__ == '__main__':
    main()
