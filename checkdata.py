import glob
import os
import shutil

import cv2


def check_data():
    annotateds = [os.path.basename(f_path) for f_path in glob.glob('/home/Downloads/Datasets/balance_output/images/train/*.jpg')]
    images = [os.path.basename(f_path) for f_path in glob.glob('/home/Downloads/Datasets/balance_output/images/test/*.jpg')]

    print(len(annotateds))
    print(len(images))

    remains = list(set(images) - set(annotateds))
    print(len(remains))

    for image_name in remains:
        img_path = os.path.join('/home/ld/Data/CarTGMT/', image_name)
        save_path = os.path.join('/home/ld/Data/remains_CarTGMT/', image_name)
        shutil.copy(img_path, save_path)


def refine_data():
    data_dir = '/home/ld/Data/yolo_data/images/train'
    source_dir = '/home/ld/Data/yolo_plate_dataset'
    for image_path in glob.glob(os.path.join(data_dir, '*.jpg')):
        img_name = os.path.basename(image_path)
        # source_image = os.path.join(source_dir, img_name)
        # if not os.path.isfile(source_image):
        #     continue
        # print(source_image)
        # shutil.copy(source_image, image_path)

        if ' (copy)' in img_name:
            source_image = os.path.join(data_dir, img_name.replace(' (copy)', ''))
            if not os.path.isfile(source_image):
                continue
            shutil.copy(source_image, image_path)


def show_results(img, xywh, landmarks):
    h, w, c = img.shape
    tl = 10 or round(0.02 * (h + w) / 2) + 1  # line/font thickness
    # tl = 5
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (102, 11, 102), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (111, 124, 0)]
    print('Number of landmark: ', len(landmarks))
    for i in range(4):
        
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        if point_x < x1 or point_x > (x2 + 5) or point_y < y1 or point_y > (y2 + 5):
            return img, False
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    return img, True


def vis_data():
    image_dir = '/home/Downloads/Datasets/balance_output/images/val/'
    label_dir = '/home/Downloads/Datasets/balance_output/labels/val/'
    save_dir = '/home/Downloads/Datasets/visual_result/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    check_list = []
    for label_path in glob.glob(os.path.join(label_dir, '*.txt')):
        label_name = os.path.basename(label_path)
        image_name = label_name.split('.txt')[0] + '.jpg'
        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(save_dir, image_name)
        img = cv2.imread(image_path)

        print(image_path)
        with open(label_path) as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                xywh = [float(i) for i in line[1:5]]
                points = [float(i) for i in line[5:]]
                vis_img, ret = show_results(img, xywh, points)
                # if not ret:
                #     check_list.append(label_path)
                # else:
                #     cv2.imwrite(save_path, vis_img)
        img = cv2.resize(img, (960, 960))
        cv2.imshow(image_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # new_label_dir = '/home/ld/Data/yolo_data/new_label3'
    # if not os.path.isdir(new_label_dir):
    #     os.makedirs(new_label_dir)
    # for label_path in check_list:
    #     print(label_path)
    #     label_name = os.path.basename(label_path)
    #     save_path = os.path.join(new_label_dir, label_name)
    #     print(save_path)
    #     shutil.copy(label_path, save_path)
    #     with open(save_path, 'w') as fw:
    #         with open(label_path) as f:
    #             for line in f.readlines():
    #                 line = line.strip().split(' ')
    #                 points = [float(i) for i in line[5:]]
    #                 save_points = []
    #                 for i in range(4):
    #                     save_points.append(points[i])
    #                     save_points.append(points[i + 4])
    #                 save_str = ' '.join(line[0:5]) + ' ' + ' '.join([str(i) for i in save_points]) + '\n'
    #                 fw.write(save_str)


def test():
    label_dir2 = '/home/ld/Data/yolo_data/new_label2'
    label_dir = '/home/ld/Data/yolo_data/new_label'
    save_dir = '/home/ld/Data/yolo_data/save_label'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    label2 = os.listdir(label_dir2)
    for label_name in os.listdir(label_dir):
        if label_name in label2:
            print(label_name)
            continue
        label_path = os.path.join(label_dir, label_name)
        save_path = os.path.join(save_dir, label_name)
        shutil.copy(label_path, save_path)



vis_data()
