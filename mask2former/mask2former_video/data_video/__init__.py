# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper
from .dataset_mapper_vspw import VSPWDatasetMapper
from .build import *

from .datasets import *
from .ytvis_eval import YTVISEvaluator
from .vspw_eval import VSPWEvaluator
