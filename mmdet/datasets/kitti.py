import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class KittiDataset(CocoDataset):

    CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc')
    def evaluate(self, results, iou_thrs = [0.5], **kwargs):
        return super().evaluate(
                results,
                iou_thrs = iou_thrs,
                **kwargs,
                )

