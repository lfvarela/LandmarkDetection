import os
import shutil
import tqdm
import pickle
import csv
import threading
from PIL import Image


OUTDIR = 'ImagesInDirs/'
NUM_WORKERS = 8
NEW_W = 224
NEW_H = 224

class Resizer():
    '''
    Resizes images from OUTDIR into (NEW_W, NEW_H)
    '''
    def resize_images(self):

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

                img_path = image_list.pop()
                c.release()
                new_file_name = '{}_R.jpg'.format(img_path[:-4])
                with Image.open(img_path) as image:
                    image = image.resize((NEW_W, NEW_H))
                    image.save(new_file_name, format='JPEG')
                os.remove(img_path)

        class_dirs = os.listdir(OUTDIR)
        image_list = []
        for class_dir in class_dirs:
            class_dir_path = os.path.join(OUTDIR, class_dir)
            imgs = os.listdir(class_dir_path)
            for img in imgs:
                image_list.append(os.path.join(OUTDIR, class_dir, img))

        global c
        c = threading.Condition()
        c.acquire()

        # Start and join threads
        threads = [ threading.Thread(target=thread_target, args=(str(i), image_list)) for i in range(NUM_WORKERS) ]

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
        self.resize_images()


def main():
    loader = Resizer()
    loader.run()


if __name__ == '__main__':
    main()
