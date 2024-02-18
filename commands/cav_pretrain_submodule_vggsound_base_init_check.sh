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
PYTHONPATH=. python main.py --config-name=cav_pretrain_submodule_vggsound -m \
environment.port=21998 \
backbone=cav \
backbone.args.mid_fusion_depth=10 \
criterion.args.load_local_path=${home_path}/STELLA_code/baseline_ckpt/vggsound_pretrained.pth \
cl_algo=finetune_submodule \
cl_algo.args.matching_loss_weight=1.0 \
data_augm=cav_augm \
data.target_task=['Origin'] \
data.skip_task=[] \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.use_audio=True \
data.args.num_frames=4 \
logging.eval_freq=1 \
logging.resume_path='' \
logging.name=cav_base_vggsound_pretrain_submodule \
logging.suffix=_finetune \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=4 \
environment.seed=2023 \
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
optim.epochs=8 \
optim.batch_size=48 \
optim.per_gpu_batchsize=12 \
optim.layer_decay=1.0 \
optim.use_lr_scheduler=False \

