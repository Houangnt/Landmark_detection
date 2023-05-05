import json
import argparse
import sys
import cv2
import os
import shutil


def main(args):
    # id_list = make_id_list(args.data_dir)
    # print(id_list)
    for json_file in os.listdir(args.data_dir):
        if json_file.split('.')[-1] == 'json':
            print(json_file)
            json_path = os.path.join(args.data_dir, json_file)
            txt_path = os.path.join(args.save_dir, os.path.basename(json_path).split('.')[0] + '.txt')
            with open(txt_path, 'w') as f:
                data = json.load(open(json_path))
                ports_list = []
                for i in range(len(data['shapes'])):
                    ports_location = data['shapes'][i]['points']
                    ports_list.append(ports_location)

                for i in range(len(data['shapes'])):
                    x_lists = []
                    y_lists = []
                    obj_ports = ports_list[i]
                    for port in obj_ports:
                        x_lists.append(port[0])
                        y_lists.append(port[1])
                    x_max = sorted(x_lists)[-1]
                    x_min = sorted(x_lists)[0]
                    y_max = sorted(y_lists)[-1]
                    y_min = sorted(y_lists)[0]

                    width = x_max - x_min
                    height = y_max - y_min
                    img_path = os.path.join(args.data_dir, json_file.split('.')[0] + '.jpg')
                    if not os.path.exists(img_path):
                        img_path = img_path.replace('.jpg', '.png')
                    u, v, _ = cv2.imread(img_path).shape
                    # u, v = data["imageHeight"], data["imageWidth"]

                    center_x, center_y = round(float((x_min + width / 2.0) / v), 6), round(
                        float((y_min + height / 2.0) / u), 6)

                    f_width, f_height = round(float(width / v), 6), round(float(height / u), 6)

                    # label_id = str(id_list[data['shapes'][i]['label']])
                    label_id = data['shapes'][i]['label']
                    print(label_id)
                    # print(data['shapes'][i]['label'])

                    str_points = ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(f_width) + ' ' + str(f_height) + ' '

                    for port in obj_ports:
                        str_points += str(port[0] / v) + ' ' + str(port[1] / u) + ' '
                    f.write(label_id  +  str_points + '\n')
                    # save_yolo_file(label_id, str(center_x), str(center_y), str(f_width), str(f_height), str_points,
                    #                 args.data_dir, json_path, args.save_dir)
                # break 
        else:
            pass

    # for file in os.listdir(args.data_dir):
    #     b_name = ['png', 'jpg']

    #     dirs_path = os.path.join(args.save_dir, 'train')
    #     dirs_path_val = os.path.join(args.save_dir, 'val')
    #     if os.path.exists(dirs_path):
    #         pass
    #     else:
    #         os.makedirs(dirs_path)
    #         os.makedirs(dirs_path_val)

    #     if not os.path.exists(os.path.join(args.data_dir, file.split('.')[0] + '.' + 'json')):
    #         continue

    #     if file.split('.')[-1] in b_name:
    #         pass
    #         # shutil.copyfile(os.path.join(args.data_dir, file), os.path.join(dirs_path, file.replace('.png', '.jpg')))

    # get_train_or_val(args.data_dir, args.save_dir)
    # get_label_id(args.data_dir, id_list, args.save_dir)
    print("save done!")

# def make_id_list(src_path):
#     json_list = []
#     id_list = []
#     for f_path in os.listdir(src_path):
#         if f_path.split('.')[-1] == 'json':
#             json_list.append(os.path.join(src_path, f_path))
#         else:
#             pass
#     for json_path in json_list:
#         data = json.load(open(json_path))
#         for i in range(len(data['shapes'])):
#             label = data['shapes'][i]['label']
#             id_list.append(label)
#     id_list = list(set(id_list))

#     index = range(len(id_list))

#     id_dict = dict(zip(id_list, index))

#     return id_dict


def save_yolo_file(id_name, x, y, w, h, points, path, json_path, save_dir):
    dir_path = os.path.join(save_dir, 'train')
    # pdb.set_trace()
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)

    txt_path = os.path.join(dir_path, os.path.basename(json_path).split('.')[0] + '.txt')

    with open(txt_path, 'a+') as f:
        f.write(id_name + ' ' + x + ' ' + y + ' ' + w + ' ' + h + ' ' + points + '\n')

    return 0


def get_train_or_val(path, save_dir):
    Txt_path = os.path.join(save_dir, 'yolo_train.txt')
    for file_name in os.listdir(path):
        if file_name.split('.')[-1] == 'jpg' or file_name.split('.')[-1] == 'png':
            img_path = os.path.join(path, file_name)
            with open(Txt_path, 'a+') as f:
                f.write(img_path + '\n')


def get_label_id(path, id_list, save_dir):
    Txt_path = os.path.join(save_dir, 'names.txt')
    with open(Txt_path, 'w') as f:
        atu = sorted(id_list.items(), key=lambda x: x[1], reverse=False)
        for ind in range(len(atu)):
            alist = list(atu[ind])
            print(alist[0])
            f.write(alist[0] + '\n')


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='the path of the json file', default='labelme_images')
    parser.add_argument('--save_dir', type=str,
                        help='the path of the json file', default='labelme_images')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
