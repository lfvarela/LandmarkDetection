
import os, sys
from PIL import Image
from io import BytesIO

def rotate():
    if len(sys.argv) != 3:
        print('Syntax: {} <input_dir/> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (input_dir, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    images = os.listdir(input_dir)
    for img_file in images:
        with Image.open(os.path.join(input_dir, img_file)) as img:
            img_rotated = img.transpose(Image.FLIP_LEFT_RIGHT)
            filename = os.path.join(out_dir, img_file[:-4] + '_rotate.jpg')
            img_rotated.save(filename, format='JPEG')

# Usage: python rotate.py Images Images-rotated
if __name__ == '__main__':
    rotate()
