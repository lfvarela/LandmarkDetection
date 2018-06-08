import numpy as np
import os
from scipy.misc import imread
import threading

NUM_WORKERS = 8
data_dir = 'TrainDatasets/Train0'

def add_one_mean(img):
    global mean_sum
    mean_sum += imread(img, mode='RGB')

def add_one_std(img):
    global sum_std
    global mean
    sum_std += (imread(img,mode='RGB') - mean)**2

def gen_path_list():
    l = []
    for subdir in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, subdir)):
            imgpath = os.path.join(data_dir, subdir, img)
            l.append(imgpath)
    return l


def process_compute_threaded(compute, l):

    def thread_target(t_id, compute, l):
        global c
        global num_done
        while True:
            c.acquire()
            remaining = len(l)
            if remaining % 10000 == 0:
                print('Images remaining: {}'.format(remaining))
            if remaining == 0:
                num_done += 1
                if num_done == NUM_WORKERS:
                    c.notify_all()
                c.release()
                return

            img_path = l.pop()
            compute(img_path)
            c.release()

    global c
    c = threading.Condition()
    c.acquire()

    # Start and join threads
    threads = [ threading.Thread(target=thread_target, args=(str(i), compute, l)) for i in range(NUM_WORKERS) ]

    global num_done
    num_done = 0
    for t_id, t in enumerate(threads):
        t.start()
    c.wait()
    c.release()
    for t_id, t in enumerate(threads):
        threads[t_id].join()

    print('DONE')


if __name__ == '__main__':
    global mean_sum
    mean_sum = np.zeros((224, 224, 3))
    global sum_std
    sum_std = np.zeros((224, 224, 3))
    l = gen_path_list()
    n = len(l)
    l2 = l.copy()
    process_compute_threaded(add_one_mean, l)
    global mean
    mean = mean_sum / n
    process_compute_threaded(add_one_std, l2)
    std = np.sqrt(sum_std / n)
    np.save('mean_{}.npy'.format(data_dir), mean)
    np.save('std_{}.npy'.format(data_dir), std)
