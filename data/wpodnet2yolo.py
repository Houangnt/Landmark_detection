import os
import argparse
import shutil
import sys


def main(args):
    if not os.path.isdir(args.label_save_dir):
        os.makedirs(args.label_save_dir)
    if not os.path.isdir(args.img_save_dir):
        os.makedirs(args.img_save_dir)
    for label_name in os.listdir(args.anno_dir):
        image_name = label_name.split('.txt')[0] + '.jpg'
        image_path = os.path.join(args.image_dir, image_name)
        label_path = os.path.join(args.anno_dir, label_name)
        if not os.path.isfile(image_path):
            continue
        with open(os.path.join(args.label_save_dir, label_name), 'w') as fw:
            with open(label_path) as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    if line[-2] == 'LP':
                        points = [float(i) for i in line[1:-2]]
                        print(points)
                        x_max = max(points[0], points[2], points[4], points[6])
                        x_min = min(points[0], points[2], points[4], points[6])
                        y_max = max(points[1], points[3], points[5], points[7])
                        y_min = min(points[1], points[3], points[5], points[7])
                        w = x_max - x_min
                        h = y_max - y_min
                        center_x = (x_max + x_min) / 2
                        center_y = (y_max + y_min) / 2
                        fw.write('{0} {1} {2} {3} {4} {5}\n'.format(0, center_x, center_y, w, h, ' '.join(line[1:-2])))
        shutil.copy(image_path, os.path.join(args.img_save_dir, image_name))


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_dir', type=str,
                        help='the path of the txt files', default='')
    parser.add_argument('--image_dir', type=str,
                        help='the path of the image file', default='')
    parser.add_argument('--img_save_dir', type=str,
                        help='the path of the image file', default='')
    parser.add_argument('--label_save_dir', type=str,
                        help='the path of the image file', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
