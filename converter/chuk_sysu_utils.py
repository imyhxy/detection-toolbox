import os.path as osp

from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
from xmltodict import unparse

from converter.coco_utils import base_dict, base_object
from converter.voc_utils import make_voc_folder

__all__ = ['chuksysu2voc']


def chuksysu2voc(root_dir, dst_dir):
    dst_dirs = make_voc_folder(dst_dir)
    anno_file = osp.join(root_dir, 'annotation', 'Images.mat')
    img_root = osp.join(root_dir, 'Image', 'SSM', '{}')
    annos = loadmat(anno_file)

    image_anns = {}
    for an in tqdm(annos['Img'][0], 'Parse SYSU Annotations'):
        img_name = an[0][0]
        img = Image.open(img_root.format(img_name))
        w, h = img.size
        img_id = osp.splitext(img_name)[0]
        img = base_dict(img_name, w, h)
        image_anns[img_id] = img

        for box, difficult in an[2][0]:
            box = box.flatten().tolist()
            size_info = {'width': w, 'height': h}
            difficult = str(difficult[0][0])
            obj = base_object(size_info, 'person', box, difficult)
            image_anns[img_id]['annotation']['object'].append(obj)

    image_anns = {k: v for k, v in image_anns.items() if len(v['annotation']['object'])}

    for k, im in tqdm(image_anns.items(), 'Write Annotations'):
        unparse(im, open(osp.join(dst_dirs['Annotations'], '{}.xml'.format(k)), 'w'),
                full_document=False, pretty=True)

    with open(osp.join(dst_dirs['ImageSets'], 'train.txt'), 'w') as f:
        f.writelines([x + '\n' for x in image_anns.keys()])


if __name__ == '__main__':
    root_dir = '/home/imy/datasets/chuk-sysu'
    dst_dir = 'debug'
    chuksysu2voc(root_dir, dst_dir)
