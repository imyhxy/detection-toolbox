import os
import os.path as osp

from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
from xmltodict import unparse

from converter.coco_utils import base_dict, base_object
from converter.voc_utils import make_voc_folder

__all__ = ['prw2voc']


def prw2voc(sor_dir, output_dir):
    dst_dirs = make_voc_folder(output_dir)
    anno_root = osp.join(sor_dir, 'annotations')
    img_root = osp.join(sor_dir, 'frames', '{}')
    anno_files = os.listdir(anno_root)

    image_anns = {}
    for an in tqdm(anno_files, 'Parse PRW Annotations'):
        img_name = osp.splitext(an)[0]
        img_id = osp.splitext(img_name)[0]
        an = loadmat(osp.join(anno_root, an))
        img = Image.open(img_root.format(img_name))
        w, h = img.size
        size_info = {'width': w, 'height': h}
        img = base_dict(img_name, w, h)
        image_anns[img_id] = img

        anno_key = [x for x in an.keys() if not x.startswith('__')]
        assert len(anno_key) == 1, img_name

        for a in an[anno_key[0]]:
            box = a[1:].tolist()
            obj = base_object(size_info, 'person', box, '0')
            image_anns[img_id]['annotation']['object'].append(obj)

    image_anns = {k: v for k, v in image_anns.items() if len(v['annotation']['object'])}

    for k, im in tqdm(image_anns.items(), 'Write Annotations'):
        unparse(im, open(osp.join(dst_dirs['Annotations'], '{}.xml'.format(k)), 'w'),
                full_document=False, pretty=True)

    with open(osp.join(dst_dirs['ImageSets'], 'train.txt'), 'w') as f:
        f.writelines([x + '\n' for x in image_anns.keys()])


if __name__ == '__main__':
    root_dir = '/home/imy/datasets/prw'
    output_dir = 'debug'
    prw2voc(root_dir, output_dir)
