import os
import shutil
import sys

def gen_mini_train(data_dir):
    for subdir in os.walk(data_dir):
        new_subdir = os.path.join('../mini-train/', subdir)
        os.makedirs(new_subdir)
        assert(os.path.isdir(new_subdir))
        for img in os.listdir(subdir)[:10]:
            if img.endswith(".jpg"):
                shutil.copy(img, new_subdir)

if __name__ == '__main__':
    directory = sys.argv[1]
    gen_mini_train(directory)