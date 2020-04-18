import os
import os.path as osp

from converter import chuksysu2voc, coco2voc, filter_voc, make_voc_folder, prw2voc, random_split_from_annotations


def link_among_folder(sor_dir, dst_dir):
    sor_dir = osp.abspath(osp.expanduser(sor_dir))
    dst_dir = osp.abspath(osp.expanduser(dst_dir))
    for f in os.listdir(sor_dir):
        try:
            os.symlink(osp.join(sor_dir, f), osp.join(dst_dir, f))
        except FileExistsError:
            pass


def link_voc(sor_dir, dst_dir):
    for f in ['Annotations', 'JPEGImages']:
        link_among_folder(osp.join(sor_dir, f), osp.join(dst_dir, f))


def _grab_voc(voc_list):
    for x, y, z in voc_list:
        filter_voc(x, y, z, filter_cats=['person'])
        link_among_folder(osp.join(x, 'JPEGImages'), osp.join(y, 'JPEGImages'))
        link_voc(y, output_dir)


def _grab_coco(coco_list):
    for x, y, z in coco_list:
        coco2voc(x, y, z, filter_cats=['person'])
        coco_root = osp.dirname(osp.dirname(x))
        coco_set = osp.basename(x).split('.')[0].split('_')[-1]
        link_among_folder(os.path.join(coco_root, coco_set), osp.join(y, 'JPEGImages'))
        link_voc(y, output_dir)


def _grab_chuk_sysu(chuk_sysu_list):
    for x, y in chuk_sysu_list:
        chuksysu2voc(x, y)
        link_among_folder(osp.join(x, 'Image', 'SSM'), osp.join(y, 'JPEGImages'))
        link_voc(y, output_dir)


def _grab_prw(prw_list):
    for x, y in prw_list:
        prw2voc(x, y)
        link_among_folder(osp.join(x, 'frames'), osp.join(y, 'JPEGImages'))
        link_voc(y, output_dir)


if __name__ == '__main__':
    root_dir = '/home/imy/datasets/VOCdevkit/Person'
    output_dir = osp.join(root_dir, 'VOCPerson')
    dst_dirs = make_voc_folder(output_dir)

    coco_list = [
        ('/home/imy/datasets/coco2017/annotations/instances_train2017.json', osp.join(root_dir, 'coco2017'), 'train'),
        ('/home/imy/datasets/coco2017/annotations/instances_val2017.json', osp.join(root_dir, 'coco2017'), 'val')]
    voc_list = [('/home/imy/datasets/VOCdevkit/VOC2007', osp.join(root_dir, 'voc2007'), 'trainval'),
                ('/home/imy/datasets/VOCdevkit/VOC2007', osp.join(root_dir, 'voc2007'), 'test'),
                ('/home/imy/datasets/VOCdevkit/VOC2012', osp.join(root_dir, 'voc2012'), 'trainval')]
    chuk_sysu_list = [('/home/imy/datasets/chuk-sysu', osp.join(root_dir, 'chuksysu'))]
    prw_list = [('/home/imy/datasets/prw', osp.join(root_dir, 'prw'))]

    grabbers = {
        'voc': lambda: _grab_voc(voc_list),
        'coco': lambda: _grab_coco(coco_list),
        'prw': lambda: _grab_prw(prw_list),
        'chuk': lambda: _grab_chuk_sysu(chuk_sysu_list)
    }

    grab_list = ['voc', 'coco', 'prw', 'chuk']
    for gl in grab_list:
        grabbers[gl]()

    random_split_from_annotations(dst_dirs['Annotations'], dst_dirs['ImageSets'])
