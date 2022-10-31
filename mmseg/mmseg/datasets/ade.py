# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADE20KDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    def __init__(self, **kwargs):
        super(ADE20KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files

@DATASETS.register_module()
class ADE20KFULLDataset(CustomDataset):
    """ADE20KFULL dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 847 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.tif'.
    """
    CLASSES = (
        'wall', 'building', 'sky', 'tree', 'road', 'floor', 'ceiling', 'bed', 
        'sidewalk', 'earth', 'cabinet', 'person', 'grass', 'windowpane', 'car', 
        'mountain', 'plant', 'table', 'chair', 'curtain', 'door', 'sofa', 'sea', 
        'painting', 'water', 'mirror', 'house', 'rug', 'shelf', 'armchair', 
        'fence', 'field', 'lamp', 'rock', 'seat', 'river', 'desk', 'bathtub', 
        'railing', 'signboard', 'cushion', 'path', 'work-surface', 'stairs', 
        'column', 'sink', 'wardrobe', 'snow', 'refrigerator', 'base', 'bridge', 
        'blind', 'runway', 'cliff', 'sand', 'fireplace', 'pillow', 'screen-door', 
        'toilet', 'skyscraper', 'grandstand', 'box', 'pool-table', 'palm', 
        'double-door', 'coffee-table', 'counter', 'countertop', 'chest-of-drawers', 
        'kitchen-island', 'boat', 'waterfall', 'stove', 'flower', 'bookcase', 
        'controls', 'book', 'stairway', 'streetlight', 'computer', 'bus', 
        'swivel-chair', 'light', 'bench', 'case', 'towel', 'fountain', 'embankment', 
        'television-receiver', 'van', 'hill', 'awning', 'poster', 'truck', 
        'airplane', 'pole', 'tower', 'court', 'ball', 'aircraft-carrier', 'buffet', 
        'hovel', 'apparel', 'minibike', 'animal', 'chandelier', 'step', 'booth', 
        'bicycle', 'doorframe', 'sconce', 'pond', 'trade-name', 'bannister', 'bag', 
        'traffic-light', 'gazebo', 'escalator', 'land', 'board', 'arcade-machine', 
        'eiderdown', 'bar', 'stall', 'playground', 'ship', 'ottoman', 'ashcan', 
        'bottle', 'cradle', 'pot', 'conveyer-belt', 'train', 'stool', 'lake', 
        'tank', 'ice', 'basket', 'manhole', 'tent', 'canopy', 'microwave', 'barrel', 
        'dirt-track', 'beam', 'dishwasher', 'plate', 'screen', 'ruins', 'washer', 
        'blanket', 'plaything', 'food', 'screen', 'oven', 'stage', 'beacon', 
        'umbrella', 'sculpture', 'aqueduct', 'container', 'scaffolding', 'hood', 
        'curb', 'roller-coaster', 'horse', 'catwalk', 'glass', 'vase', 
        'central-reservation', 'carousel', 'radiator', 'closet', 'machine', 'pier', 
        'fan', 'inflatable-bounce-game', 'pitch', 'paper', 'arcade', 'hot-tub', 
        'helicopter', 'tray', 'partition', 'vineyard', 'bowl', 'bullring', 'flag', 
        'pot', 'footbridge', 'shower', 'bag', 'bulletin-board', 'confessional-booth', 
        'trunk', 'forest', 'elevator-door', 'laptop', 'instrument-panel', 'bucket', 
        'tapestry', 'platform', 'jacket', 'gate', 'monitor', 'telephone-booth', 
        'spotlight', 'ring', 'control-panel', 'blackboard', 'air-conditioner', 
        'chest', 'clock', 'sand-dune', 'pipe', 'vault', 'table-football', 'cannon', 
        'swimming-pool', 'fluorescent', 'statue', 'loudspeaker', 'exhibitor', 
        'ladder', 'carport', 'dam', 'pulpit', 'skylight', 'water-tower', 'grill', 
        'display-board', 'pane', 'rubbish', 'ice-rink', 'fruit', 'patio', 
        'vending-machine', 'telephone', 'net', 'backpack', 'jar', 'track', 
        'magazine', 'shutter', 'roof', 'banner', 'landfill', 'post', 'altarpiece', 
        'hat', 'arch', 'table-game', 'bag', 'document', 'dome', 'pier', 'shanties', 
        'forecourt', 'crane', 'dog', 'piano', 'drawing', 'cabin', 'ad', 'amphitheater', 
        'monument', 'henhouse', 'cockpit', 'heater', 'windmill', 'pool', 'elevator', 
        'decoration', 'labyrinth', 'text', 'printer', 'mezzanine', 'mattress', 'straw', 
        'stalls', 'patio', 'billboard', 'bus-stop', 'trouser', 'console-table', 'rack', 
        'notebook', 'shrine', 'pantry', 'cart', 'steam-shovel', 'porch', 'postbox', 
        'figurine', 'recycling-bin', 'folding-screen', 'telescope', 'deck-chair', 
        'kennel', 'coffee-maker', 'altar', 'fish', 'easel', 'artificial-golf-green', 
        'iceberg', 'candlestick', 'shower-stall', 'television-stand', 'wall-socket', 
        'skeleton', 'grand-piano', 'candy', 'grille-door', 'pedestal', 'jersey', 
        'shoe', 'gravestone', 'shanty', 'structure', 'rocking-chair', 'bird', 
        'place-mat', 'tomb', 'big-top', 'gas-pump', 'lockers', 'cage', 'finger', 
        'bleachers', 'ferris-wheel', 'hairdresser-chair', 'mat', 'stands', 'aquarium', 
        'streetcar', 'napkin', 'dummy', 'booklet', 'sand-trap', 'shop', 'table-cloth', 
        'service-station', 'coffin', 'drawer', 'cages', 'slot-machine', 'balcony', 
        'volleyball-court', 'table-tennis', 'control-table', 'shirt', 'merchandise', 
        'railway', 'parterre', 'chimney', 'can', 'tanks', 'fabric', 'alga', 'system', 
        'map', 'greenhouse', 'mug', 'barbecue', 'trailer', 'toilet-tissue', 'organ', 
        'dishrag', 'island', 'keyboard', 'trench', 'basket', 'steering-wheel', 'pitcher', 
        'goal', 'bread', 'beds', 'wood', 'file-cabinet', 'newspaper', 'motorboat', 
        'rope', 'guitar', 'rubble', 'scarf', 'barrels', 'cap', 'leaves', 'control-tower', 
        'dashboard', 'bandstand', 'lectern', 'switch', 'baseboard', 'shower-room', 
        'smoke', 'faucet', 'bulldozer', 'saucepan', 'shops', 'meter', 'crevasse', 'gear', 
        'candelabrum', 'sofa-bed', 'tunnel', 'pallet', 'wire', 'kettle', 'bidet', 
        'baby-buggy', 'music-stand', 'pipe', 'cup', 'parking-meter', 'ice-hockey-rink', 
        'shelter', 'weeds', 'temple', 'patty', 'ski-slope', 'panel', 'wallet', 'wheel', 
        'towel-rack', 'roundabout', 'canister', 'rod', 'soap-dispenser', 'bell', 
        'canvas', 'box-office', 'teacup', 'trellis', 'workbench', 'valley', 'toaster', 
        'knife', 'podium', 'ramp', 'tumble-dryer', 'fireplug', 'gym-shoe', 'lab-bench', 
        'equipment', 'rocky-formation', 'plastic', 'calendar', 'caravan', 
        'check-in-desk', 'ticket-counter', 'brush', 'mill', 'covered-bridge', 
        'bowling-alley', 'hanger', 'excavator', 'trestle', 'revolving-door', 
        'blast-furnace', 'scale', 'projector', 'soap', 'locker', 'tractor', 'stretcher', 
        'frame', 'grating', 'alembic', 'candle', 'barrier', 'cardboard', 'cave', 'puddle', 
        'tarp', 'price-tag', 'watchtower', 'meters', 'light-bulb', 'tracks', 'hair-dryer', 
        'skirt', 'viaduct', 'paper-towel', 'coat', 'sheet', 'fire-extinguisher', 
        'water-wheel', 'pottery', 'magazine-rack', 'teapot', 'microphone', 'support', 
        'forklift', 'canyon', 'cash-register', 'leaf', 'remote-control', 'soap-dish', 
        'windshield', 'cat', 'cue', 'vent', 'videos', 'shovel', 'eaves', 'antenna', 
        'shipyard', 'hen', 'traffic-cone', 'washing-machines', 'truck-crane', 'cds', 
        'niche', 'scoreboard', 'briefcase', 'boot', 'sweater', 'hay', 'pack', 
        'bottle-rack', 'glacier', 'pergola', 'building-materials', 'television-camera', 
        'first-floor', 'rifle', 'tennis-table', 'stadium', 'safety-belt', 'cover', 
        'dish-rack', 'synthesizer', 'pumpkin', 'gutter', 'fruit-stand', 'ice-floe', 
        'handle', 'wheelchair', 'mousepad', 'diploma', 'fairground-ride', 'radio', 
        'hotplate', 'junk', 'wheelbarrow', 'stream', 'toll-plaza', 'punching-bag', 
        'trough', 'throne', 'chair-desk', 'weighbridge', 'extractor-fan', 
        'hanging-clothes', 'dish', 'alarm-clock', 'ski-lift', 'chain', 'garage', 
        'mechanical-shovel', 'wine-rack', 'tramway', 'treadmill', 'menu', 'block', 'well', 
        'witness-stand', 'branch', 'duck', 'casserole', 'frying-pan', 'desk-organizer', 
        'mast', 'spectacles', 'service-elevator', 'dollhouse', 'hammock', 
        'clothes-hanging', 'photocopier', 'notepad', 'golf-cart', 'footpath', 'cross', 
        'baptismal-font', 'boiler', 'skip', 'rotisserie', 'tables', 'water-mill', 
        'helmet', 'cover-curtain', 'brick', 'table-runner', 'ashtray', 'street-box', 
        'stick', 'hangers', 'cells', 'urinal', 'centerpiece', 'portable-fridge', 'dvds', 
        'golf-club', 'skirting-board', 'water-cooler', 'clipboard', 'camera', 
        'pigeonhole', 'chips', 'food-processor', 'post-box', 'lid', 'drum', 'blender', 
        'cave-entrance', 'dental-chair', 'obelisk', 'canoe', 'mobile', 'monitors', 
        'pool-ball', 'cue-rack', 'baggage-carts', 'shore', 'fork', 'paper-filer', 
        'bicycle-rack', 'coat-rack', 'garland', 'sports-bag', 'fish-tank', 
        'towel-dispenser', 'carriage', 'brochure', 'plaque', 'stringer', 'iron', 
        'spoon', 'flag-pole', 'toilet-brush', 'book-stand', 'water-faucet', 
        'ticket-office', 'broom', 'dvd', 'ice-bucket', 'carapace', 'tureen', 'folders', 
        'chess', 'root', 'sewing-machine', 'model', 'pen', 'violin', 'sweatshirt', 
        'recycling-materials', 'mitten', 'chopping-board', 'mask', 'log', 'mouse', 
        'grill', 'hole', 'target', 'trash-bag', 'chalk', 'sticks', 'balloon', 'score', 
        'hair-spray', 'roll', 'runner', 'engine', 'inflatable-glove', 'games', 
        'pallets', 'baskets', 'coop', 'dvd-player', 'rocking-horse', 'buckets', 
        'bread-rolls', 'shawl', 'watering-can', 'spotlights', 'post-it', 'bowls', 
        'security-camera', 'runner-cloth', 'lock', 'alarm', 'side', 'roulette', 'bone', 
        'cutlery', 'pool-balls', 'wheels', 'spice-rack', 'plant-pots', 'towel-ring', 
        'bread-box', 'video', 'funfair', 'breads', 'tripod', 'ironing-board', 'skimmer', 
        'hollow', 'scratching-post', 'tricycle', 'file-box', 'mountain-pass', 
        'tombstones', 'cooker', 'card-game', 'golf-bag', 'towel-paper', 'chaise-lounge', 
        'sun', 'toilet-paper-holder', 'rake', 'key', 'umbrella-stand', 'dartboard', 
        'transformer', 'fireplace-utensils', 'sweatshirts', 'cellular-telephone', 
        'tallboy', 'stapler', 'sauna', 'test-tube', 'palette', 'shopping-carts', 'tools', 
        'push-button', 'star', 'roof-rack', 'barbed-wire', 'spray', 'ear', 'sponge', 
        'racket', 'tins', 'eyeglasses', 'file', 'scarfs', 'sugar-bowl', 'flip-flop', 
        'headstones', 'laptop-bag', 'leash', 'climbing-frame', 'suit-hanger', 
        'floor-spotlight', 'plate-rack', 'sewer', 'hard-drive', 'sprinkler', 'tools-box', 
        'necklace', 'bulbs', 'steel-industry', 'club', 'jack', 'door-bars', 
        'control-panel', 'hairbrush', 'napkin-holder', 'office', 'smoke-detector', 
        'utensils', 'apron', 'scissors', 'terminal', 'grinder', 'entry-phone', 
        'newspaper-stand', 'pepper-shaker', 'onions', 'central-processing-unit', 'tape', 
        'bat', 'coaster', 'calculator', 'potatoes', 'luggage-rack', 'salt', 
        'street-number', 'viewpoint', 'sword', 'cd', 'rowing-machine', 'plug', 'andiron', 
        'pepper', 'tongs', 'bonfire', 'dog-dish', 'belt', 'dumbbells', 
        'videocassette-recorder', 'hook', 'envelopes', 'shower-faucet', 'watch', 
        'padlock', 'swimming-pool-ladder', 'spanners', 'gravy-boat', 'notice-board', 
        'trash-bags', 'fire-alarm', 'ladle', 'stethoscope', 'rocket', 'funnel', 
        'bowling-pins', 'valve', 'thermometer', 'cups', 'spice-jar', 'night-light', 
        'soaps', 'games-table', 'slotted-spoon', 'reel', 'scourer', 'sleeping-robe', 
        'desk-mat', 'dumbbell', 'hammer', 'tie', 'typewriter', 'shaker', 'cheese-dish', 
        'sea-star', 'racquet', 'butane-gas-cylinder', 'paper-weight', 'shaving-brush', 
        'sunglasses', 'gear-shift', 'towel-rail', 'adding-machine')

    PALETTE = None

    def __init__(self, **kwargs):
        super(ADE20KFULLDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            ignore_index=-1,
            reduce_zero_label=False,
            int16=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            tif_filename = osp.join(imgfile_prefix, f'{basename}.tif')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint16))
            output.save(tif_filename)
            result_files.append(tif_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files

