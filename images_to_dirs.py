import os
import shutil
import tqdm
import pickle
import csv
import threading


CSV_FILE = 'CSV-files/train.csv'
OUTDIR = 'ImagesInDirs/'
IMAGES = 'Images/'
NUM_WORKERS = 8

class Loader():
    def __init__(self):
        self.id_to_lid = None     # Maps { img_id: l_id }
        self.lid_to_imgs = None   # Maps { l_id: [ img_ids with l_id ]}
        self.total_images = None  # Total number of labels in downloaded images.

    def load_lid_to_imgs(self):
        '''
        Load file to global lid_to_imgs from pickle file or creates it and makes a pickle.
        '''
        # Load or create lid_to_imgs
        print('Loading lid_to_imgs.')
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

    def generate_id_to_lid(self):
        '''
        Returns dict: labels_tuple: list of (index, l_id, l_id_count, label). Index unique id from 0 to 14950 (14951 total), where label is upsample_{1-5} or downsample
        '''

        print('Generating id_to_lid.')
        downloaded_imgs = set(os.listdir(IMAGES))
        self.total_images = len(downloaded_imgs)

        self.id_to_lid = {}
        with open(CSV_FILE) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_id = row['id']
                img_file = '{}.jpg'.format(img_id)
                if img_file in downloaded_imgs:
                    self.id_to_lid[img_id] = row['landmark_id']


    def assert_num_files(self):
        print('Making sure all files were copied.')
        total_copies = 0
        class_dirs = os.listdir(OUTDIR)
        for class_dir in class_dirs:
            class_dir_path = os.path.join(OUTDIR, class_dir)
            imgs_in_class = os.listdir(class_dir_path)
            total_copies += len(imgs_in_class)
        assert(self.total_images == total_copies)

    def copy_images(self):

        def thread_target(t_id, image_list):

            global c
            global num_done
            while True:
                c.acquire()
                remaining = len(image_list)
                if remaining % 10000 == 0:
                    print('Images remaining: {}'.format(remaining))
                if remaining == 0:
                    num_done += 1
                    if num_done == NUM_WORKERS:
                        c.notify_all()
                    c.release()
                    return

                img = image_list.pop()
                c.release()
                img_path = os.path.join(IMAGES, img)
                img_id = img[:-4]
                l_id = self.id_to_lid[img_id]
                c.acquire()
                if not os.path.exists(os.path.join(OUTDIR, str(l_id))):
                    os.mkdir(os.path.join(OUTDIR, str(l_id)))
                c.release()
                dest_path = os.path.join(OUTDIR, str(l_id), img)
                if not os.path.isfile(dest_path):
                    shutil.copyfile(img_path, dest_path)

        class_dirs = os.listdir(IMAGES)
        print(len(class_dirs))

        global c
        c = threading.Condition()
        c.acquire()

        # Start and join threads
        threads = [ threading.Thread(target=thread_target, args=(str(i), class_dirs)) for i in range(NUM_WORKERS) ]

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

        # Load pickle file: lid_to_imgs (maps { l_lid to list of images with that label }
        self.load_lid_to_imgs()
        self.generate_id_to_lid()
        self.copy_images()
        self.assert_num_files()
        print('DONE!')


def main():
    loader = Loader()
    loader.run()


if __name__ == '__main__':
    main()
