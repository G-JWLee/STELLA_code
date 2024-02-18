#!/bin/bash

export home_path="/base_path"
export home_data_path="/dataset_path"

if [ "$1" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is 0,1,2,3 by default"
  gpus="0,1,2,3"
else
  echo "CUDA_VISIBLE_DEVICES is $1"
  gpus=$1
fi

if [ "$2" == "" ]; then
  echo "Pretrain_path seed is not specified"
  pretrain_path=""
else
  echo "$2 is given as pretrain_path"
  pretrain_path=$2
fi

if [ "$3" == "" ]; then
  echo "Random seed is not specified, default is 2023"
  random_seed=2023
else
  echo "$3 is given as random seed"
  random_seed=$3
fi

export CUDA_VISIBLE_DEVICES=${gpus}
PYTHONPATH=. python main.py --config-name=cav_pretrain_msrvtt -m \
backbone=cav \
backbone.args.mask_ratio_a=0.8 \
backbone.args.mask_ratio_v=0.8 \
criterion=fa_mae_cont \
criterion.args.norm_pix_loss=True \
criterion.args.load_local_path=${home_path}/STELLA_code/experiments/checkpoints/cav_base_audioset_pretrain/cav_base_audioset_pretrain_${pretrain_path}/music/model_checkpoint_0015.pth \
criterion.args.get_va_recall_metric=True \
criterion.args.contrast_loss_weight=0.1 \
cl_algo=finetune \
data_augm=cav_augm \
data.target_task=['All'] \
data.skip_task=[] \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.use_audio=True \
data.args.num_frames=4 \
logging.eval_freq=10 \
logging.retrieve_freq=10 \
logging.name=cav_base_audioset_msrvtt_pretrain \
logging.suffix=_${pretrain_path} \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=100 \
environment.seed=${random_seed} \
environment.workers=32 \
environment.slurm=False \
environment.world_size=1 \
environment.ngpu=4 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.dist_url=env:// \
environment.rank=-1 \
optim=adam \
optim.args.lr=1e-4 \
optim.args.betas=[0.95,0.999] \
optim.args.weight_decay=5e-7 \
optim.epochs=15 \
optim.batch_size=48 \
optim.per_gpu_batchsize=12 \
optim.layer_decay=1.0 \
optim.use_lr_scheduler=True \
optim.lr_scheduler=WarmupCosineSchedule \
optim.lr_scheduler_args.final_lr_ratio=0. \

