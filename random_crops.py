import random, os, sys
from PIL import Image

NEW_H = 224
NEW_W = 224

def random_crops():
    if len(sys.argv) != 4:
        print('Syntax: {} <input_dir/> <output_dir/> <num_crops>'.format(sys.argv[0]))
        sys.exit(0)
    (input_dir, out_dir, num_crops) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    images = os.listdir(input_dir)
    for img_file in images:
        with Image.open(os.path.join(input_dir, img_file)) as img:
            for i in range(int(num_crops)):
                w, h = img.size
                filename = os.path.join(out_dir, img_file[:-4] + '_crop_{}.jpg'.format(i))
                x = random.randint(0, w-NEW_W-1)
                y = random.randint(0, h-NEW_H-1)
                img.crop((x,y, x+NEW_W, y+NEW_H)).save(filename, format='JPEG')

# Usage: python random_crops.py Images-test/Final/ Images-test/RandomCrops/ 4
if __name__ == '__main__':
    random_crops()
