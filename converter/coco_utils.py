import os

from pycocotools.coco import COCO
from tqdm import tqdm
from xmltodict import unparse

from .voc_utils import make_voc_folder

__all__ = ['coco2voc']

BBOX_OFFSET = 1


def base_dict(filename, width, height, depth=3):
    return {
        'annotation': {
            'filename': os.path.split(filename)[-1],
            'folder': 'unknown', 'segmented': '0', 'owner': {'name': 'unknown'},
            'source': {'database': 'unknown', 'annotation': 'unknown', 'image': 'unknown'},
            'size': {'width': width, 'height': height, 'depth': depth},
            'object': []
        }
    }


def base_object(size_info, name, bbox, difficult):
    x1, y1, w, h = bbox
    if w < 13 or w / h > 2:
        difficult = '1'

    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': difficult,
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }


def coco2voc(ann_file, output_dir, split, filter_cats=None):
    dst_dirs = make_voc_folder(output_dir)

    coco = COCO(ann_file)
    cate = {x: coco.loadCats(x)[0]['name'] for x in coco.getCatIds()}
    filter_cats = cate.values() if filter_cats is None else filter_cats
    img_ids = coco.getImgIds()
    image_anns = {}
    for ii in tqdm(img_ids, 'Parse Images'):
        im = coco.loadImgs(ii)[0]
        img = base_dict(im['coco_url'], im['width'], im['height'], 3)
        image_anns[im['id']] = img

    for ii in tqdm(img_ids, 'Parse Annotations'):
        ann_ids = coco.getAnnIds(ii)
        anns = coco.loadAnns(ann_ids)
        for an in anns:
            if cate[an['category_id']] in filter_cats:
                ann = base_object(image_anns[an['image_id']]['annotation']['size'], cate[an['category_id']], an['bbox'],
                                  an['iscrowd'])
                image_anns[an['image_id']]['annotation']['object'].append(ann)

    image_anns = {k: v for k, v in image_anns.items() if len(v['annotation']['object'])}

    for k, im in tqdm(image_anns.items(), 'Write Annotations'):
        unparse(im, open(os.path.join(dst_dirs['Annotations'], '{}.xml'.format(str(k).zfill(12))), 'w'),
                full_document=False, pretty=True)

    with open(os.path.join(dst_dirs['ImageSets'], '{}.txt'.format(split)), 'w') as f:
        f.writelines(list(map(lambda x: '{:012d}\n'.format(x), image_anns.keys())))


if __name__ == '__main__':
    coco2voc('/home/imy/datasets/coco2017/annotations/instances_val2017.json', 'debug', 'val', filter_cats=['person'])
