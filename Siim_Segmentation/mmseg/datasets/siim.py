from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SIIMDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    CLASSES = (
        'background', 'front')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(SIIMDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            classes = ('background','front'),
            palette = [[120, 120, 120], [6, 230, 230]],
            **kwargs)
