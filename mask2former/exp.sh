export DETECTRON2_DATASETS=/luoxiao_storage/dataset/
export CUDA_LAUNCH_BLOCKING=1
python ./train_net_video.py --num-gpus 8 --config-file $1 --resume
