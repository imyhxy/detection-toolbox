from converter.voc_utils import random_display_voc

if __name__ == '__main__':
    # check_voc('/home/imy/datasets/VOCdevkit/Person/VOCCOCO')
    random_display_voc('/home/imy/datasets/VOCdevkit/Person/coco2017', 100, with_difficult=False)
