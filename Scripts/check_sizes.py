import os
from PIL import Image
import tqdm

NUM_DATASETS = 5
OUTDIR = 'TrainDatasets/'
DATASET_DIR = 'Train'

print('Making sure all images are 224x224.')
for i in tqdm.tqdm(range(NUM_DATASETS)):
    class_dirs = os.listdir('TrainDatasets/Train{}'.format(str(i)))
    for class_dir in class_dirs:
        class_dir_path = os.path.join(OUTDIR, DATASET_DIR + str(i), class_dir)
        imgs = os.listdir(class_dir_path)
        for img in imgs:
            img_file = os.path.join(class_dir_path, img)
            with Image.open(img_file) as image:
                w, h = image.size
                assert(w == 224 and h == 224)
print('DONE')
