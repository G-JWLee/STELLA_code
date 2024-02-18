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

export CUDA_VISIBLE_DEVICES=${gpus}
PYTHONPATH=. python main.py --config-name=cav_finetune_vggsound -m \
train_algo.main_worker=run \
backbone=cav \
backbone.args.mid_fusion_depth=10 \
data_augm=cav_augm \
data_augm.audio_data.args.freqm=48 \
data_augm.audio_data.args.timem=192 \
data_augm.audio_data.args.noise=True \
criterion=va_cls \
criterion.args.load_local_path=${home_path}/STELLA_code/baseline_ckpt/vggsound_pretrained.pth \
data.target_task=['All'] \
data.args.use_audio=True \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.num_frames=4 \
logging.eval_freq=1 \
logging.name=cav_base_audioset20k_pretrain \
logging.suffix='' \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=8 \
logging.resume_path='' \
environment.workers=20 \
environment.slurm=False \
environment.world_size=1 \
environment.ngpu=4 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.dist_url=env:// \
environment.rank=-1 \
optim=adamw \
optim.layer_decay=0.75 \
optim.args.lr=5e-4 \
optim.args.betas=[0.95,0.999] \
optim.args.weight_decay=5e-6 \
optim.epochs=8 \
optim.batch_size=48 \
optim.per_gpu_batchsize=3 \
optim.use_lr_scheduler=True \
optim.lr_scheduler=WarmupCosineSchedule \
optim.lr_scheduler_args.warmup_epochs=1 \
optim.lr_scheduler_args.final_lr_ratio=0. \
