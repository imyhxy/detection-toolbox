import os
import os.path as osp
import random
from random import shuffle

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from xmltodict import parse, unparse

__all__ = ['random_split_from_annotations', 'make_voc_folder', 'filter_voc', 'check_voc']


def random_split_from_annotations(anno_dir, output_dir, split_size=(0.5, 0.25, 0.25)):
    file_list = os.listdir(anno_dir)
    file_num = len(file_list)
    split_num = [int(file_num * x) for x in split_size]
    random.shuffle(file_list)

    dataset = {
        'trainval': file_list[:split_num[0] + split_num[1]],
        'train': file_list[:split_num[0]],
        'val': file_list[split_num[0]:split_num[0] + split_num[1]],
        'test': file_list[split_num[0] + split_num[1]:]
    }

    for k, v in dataset.items():
        with open(osp.join(output_dir, '{:s}.txt'.format(k)), 'w') as f:
            f.writelines([osp.splitext(x)[0] + '\n' for x in v])


def _check_anno(anno_dir, img_dir, anno_name):
    anno = parse(open(osp.join(anno_dir, anno_name)).read())
    img_path = img_dir.format(anno['annotation']['filename'])
    assert osp.isfile(img_path), img_path
    img = Image.open(img_path)
    w, h = img.size
    assert w == int(anno['annotation']['size']['width'])
    assert h == int(anno['annotation']['size']['height'])
    assert len(anno['annotation']['object']), anno_name

    return anno, img


def check_voc(dataset_root):
    anno_dir = osp.join(dataset_root, 'Annotations')
    img_dir = osp.join(dataset_root, 'JPEGImages', '{}')

    anno_list = os.listdir(anno_dir)
    for anno_name in tqdm(anno_list, 'Checking dataset'):
        _check_anno(anno_dir, img_dir, anno_name)


def random_display_voc(dataset_root, max_num=100, with_difficult=True):
    anno_dir = osp.join(dataset_root, 'Annotations')
    img_dir = osp.join(dataset_root, 'JPEGImages', '{}')

    anno_list = os.listdir(anno_dir)
    shuffle(anno_list)
    for idx, anno_name in tqdm(enumerate(anno_list), 'Checking dataset'):
        if idx >= max_num:
            break

        anno, img = _check_anno(anno_dir, img_dir, anno_name)
        draw = ImageDraw.Draw(img)
        if isinstance(anno['annotation']['object'], dict):
            anno['annotation']['object'] = [anno['annotation']['object']]

        for obj in anno['annotation']['object']:
            if not with_difficult and int(obj['difficult']):
                continue

            box = obj['bndbox']
            x1, y1, x2, y2 = map(lambda x: int(float(x)), [box['xmin'], box['ymin'], box['xmax'], box['ymax']])
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        plt.close('all')
        plt.figure()
        plt.imshow(img)
        plt.show()


def make_voc_folder(root_dir):
    dst_dirs = {x: os.path.join(root_dir, x) for x in ['Annotations', 'ImageSets', 'JPEGImages']}
    dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], 'Main')
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)

    return dst_dirs


def filter_voc(devpath, output_dir, split, filter_cats):
    dst_dirs = make_voc_folder(output_dir)

    anno_dir = os.path.join(devpath, 'Annotations')
    split_file = os.path.join(devpath, 'ImageSets', 'Main', f'{split}.txt')

    with open(split_file) as f:
        lines = [l.strip() for l in f.readlines()]

    save_imgs = []
    for l in tqdm(lines, f'Parse {split}'):
        anno_file = os.path.join(anno_dir, f'{l}.xml')
        dictxml = parse(open(anno_file).read())

        save_objs = []
        objs = dictxml['annotation']['object']
        if isinstance(objs, list):
            for obj in dictxml['annotation']['object']:
                if obj['name'] in filter_cats:
                    save_objs.append(obj)
        else:
            if objs['name'] in filter_cats:
                save_objs.append(objs)

        if len(save_objs) == 0:
            continue
        else:
            save_imgs.append(l)
            dictxml['annotation']['object'] = save_objs
            unparse(dictxml, open(os.path.join(dst_dirs['Annotations'], '{}.xml'.format(str(l))), 'w'),
                    full_document=False, pretty=True)

        with open(os.path.join(dst_dirs['ImageSets'], f'{split}.txt'), 'w') as f:
            f.writelines([x + '\n' for x in save_imgs])


if __name__ == '__main__':
    filter_voc('/home/imy/datasets/VOCdevkit/VOC2012', 'debug', 'train', ['person'])
