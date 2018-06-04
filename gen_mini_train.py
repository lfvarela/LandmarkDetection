import os
import shutil
import sys

def gen_mini_train(data_dir):
    dirs = [s[0] for s in os.walk(data_dir)]
    for subdir in dirs:
        new_subdir = os.path.join('../mini-train/', subdir)
        os.makedirs(new_subdir)
        assert(os.path.isdir(new_subdir))
        for img in os.listdir(subdir)[:10]:
            if img.endswith(".jpg"):
                shutil.copy(os.path.join(subdir, img), new_subdir)

if __name__ == '__main__':
    directory = sys.argv[1]
    gen_mini_train(directory)
