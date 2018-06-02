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
TEMP = 'Temp/'
IMAGES = 'Images/'
DATASET_DIR = 'Train' # Dont add the '/'
OUTDIR = 'TrainDatasets/'
NUM_DATASETS = 2 # TODO: Change to 5
NUM_WORKERS = 4 # Change to 24
NEW_H = 224
NEW_W = 224

class Loader():
    def __init__(self):
        self.id_to_lid = None
        self.lid_to_imgs = None
        self.train_data = None  # TODO delete
        self.total_labels = None
        self.datasets = None
        self.label_to_index_dict = {}

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
            self.label_to_index_dict[l_id] = index
        return labels_tuples


    def transformImages(self, l_id, id_to_transformations):
        '''
        Transforms images and adds them to corresponding directories.
        '''
        img_num = 0
        index = self.label_to_index_dict[l_id]
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
                    new_img = new_img.resize((NEW_W, NEW_H)).convert(mode='RGB')
                    for i in range(NUM_DATASETS):
                        self.datasets[i][index, img_num, :] = np.array(new_img)
                    img_num += 1

    def upsample_1_to_60(self, l_id):
        '''
        Apply transformations to a single image to bring it up to 60 samples.
        flip (x2), dim/bright (x3), color-transform (x2), 4-crops (x5), total: x60
        Save the image in all five datasets.
        '''
        assert(len(self.lid_to_imgs[l_id]) == 1)
        img_id = self.lid_to_imgs[l_id][0]
        img_file = os.path.join(IMAGES, '{}.jpg'.format(img_id))


        img_num = 0 # Goes up to 60
        index = self.label_to_index_dict[l_id]

        try:
            if os.path.isfile(img_file):

                os.mkdir(os.path.join(TEMP, l_id))

                # Transpose
                with Image.open(img_file) as img:
                    transpose = img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_file_name = os.path.join(TEMP,l_id,'{}_{}.jpg'.format(img_id, ''))
                    new_file_name_t = os.path.join(TEMP, l_id,'{}_{}.jpg'.format(img_id, 't'))
                    img.save(new_file_name, format='JPEG')
                    transpose.save(new_file_name_t, format='JPEG')
                    for i in range(NUM_DATASETS):
                        self.datasets[i][index, img_num, :] = np.array(img.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                        self.datasets[i][index, img_num+1, :] = np.array(transpose.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                    img_num += 2

                # Dim/Bright
                class_dir = os.path.join(TEMP,l_id)
                new_imgs = os.listdir(class_dir)
                for img_file in new_imgs:
                    img_file_path = os.path.join(TEMP, l_id,img_file)
                    with Image.open(img_file_path) as img:
                        dim = img.point(lambda p: p * 0.7)
                        brighten = img.point(lambda p: p * 1.3)
                        new_file_name_d = os.path.join(TEMP, l_id, '{}{}.jpg'.format(img_file[:-4], 'd'))
                        new_file_name_b = os.path.join(TEMP,l_id, '{}{}.jpg'.format(img_file[:-4], 'b'))
                        dim.save(new_file_name_d, format='JPEG')
                        brighten.save(new_file_name_b, format='JPEG')
                        for i in range(NUM_DATASETS):
                            self.datasets[i][index, img_num, :] = np.array(dim.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                            self.datasets[i][index, img_num+1, :] = np.array(brighten.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                        img_num += 2

                new_imgs = os.listdir(class_dir)
                for img_file in new_imgs:
                    img_file_path = os.path.join(TEMP, l_id,img_file)
                    with Image.open(img_file_path) as img:
                        grey = img.convert('L')
                        new_file_name_g = os.path.join(TEMP,l_id, '{}{}.jpg'.format(img_file[:-4], 'g'))
                        for i in range(NUM_DATASETS):
                            self.datasets[i][index, img_num, :] = np.array(grey.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                        img_num += 1

                new_imgs = os.listdir(class_dir)
                for img_file in new_imgs:
                    img_file_path = os.path.join(TEMP, l_id,img_file)
                    with Image.open(img_file_path) as img:
                        w, h = img.size
                        for j in range(4):
                            x = random.randint(0, w-NEW_W-1)
                            y = random.randint(0, h-NEW_H-1)
                            crop = img.crop((x,y, x+NEW_W, y+NEW_H))
                            for i in range(NUM_DATASETS):
                                self.datasets[i][index, img_num, :] = np.array(crop.resize((NEW_W, NEW_H)).convert(mode='RGB'))
                            img_num += 1

                fileList = os.listdir(os.path.join(TEMP, l_id))
                for fileName in fileList:
                    os.remove(os.path.join(TEMP,l_id, fileName))
                os.rmdir(os.path.join(TEMP, l_id))

            else:
                return
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
            '''

            (index, l_id, lid_count, label) = img_label

            # For downsampling: for every dataset, sample 60 random images that have the label l_id
            if label == 'downsample':
                index = self.label_to_index_dict[l_id]
                for i in range(NUM_DATASETS):
                    downsamples = np.random.choice(self.lid_to_imgs[l_id], 60, replace=False)
                    for img_num, img_id in enumerate(downsamples):
                        img_file_path = os.path.join(IMAGES, '{}.jpg'.format(img_id))
                        with Image.open(img_file_path) as img:
                            self.datasets[i][index, img_num, :] = np.array(img.resize((NEW_W, NEW_H)).convert(mode='RGB'))

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

        if not os.path.exists(TEMP):
            os.mkdir(TEMP)

        def thread_target(t_id):
            global c
            global num_done
            while True:
                c.acquire()
                remaining = len(labels)
                if remaining % 1000 == 0:
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

        # Wait for threads to be done
        c.wait()
        c.release()
        for t_id, t in enumerate(threads):
            threads[t_id].join()

        print('DONE')


    def run(self):

        labels = self.get_labels(CSV_FILE)  # Produce labels. maps { l_id: label_of(l_id) }
        self.total_labels = len(labels)
        print('num labels: {}'.format(self.total_labels))

        self.datasets = [ np.zeros((self.total_labels, 60, NEW_H, NEW_W, 3), dtype=np.uint8)] * NUM_DATASETS

        # Load pickle file: lid_to_imgs (maps { l_lid to list of images with that label }
        self.load_lid_to_imgs()

        # Load
        self.process_labels_threaded(labels)

        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

        self.classes = np.zeros(self.total_labels*60, dtype=np.int64))
        for l_id, index in self.label_to_index_dict.items():
            self.classes[index] = l_id
        self.classes =  np.outer(self.classes, np.ones(60)).reshape(len(self.classes)*60) # shape (N), each one represents the class for img n.
        np.save(os.path.join(OUTDIR, 'classes.npy'), self.classes)

        try:

            for i in range(NUM_DATASETS):
                self.datasets[i] = self.datasets[i].reshape((self.total_labels*60, NEW_H, NEW_W, 3))
                np.save(os.path.join(OUTDIR, 'train{}.npy'.format(i)), self.datasets[i])

        except Exception as e:
            print('Unable to process l_id: {}, label: {}. Error: {}'.format(l_id, label, e))
            print(traceback.format_exc())
            return 1



def main():
    loader = Loader()
    loader.run()


if __name__ == '__main__':
    main()
