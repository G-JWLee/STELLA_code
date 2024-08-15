# STELLA

### **STELLA: Continual Audio-Video Pre-training with Spatio-Temporal Localized Alignment**  
* Authors: [Jaewoo Lee*](https://g-jwlee.github.io/), [Jaehong Yoon*](https://jaehong31.github.io/), [Wonjae Kim](https://wonjae.kim/), [Yunji Kim](https://github.com/YunjiKim), [Sung Ju Hwang](http://www.sungjuhwang.com/)
* [Paper](https://arxiv.org/abs/2310.08204)
* [Project Page](https://cl-stella.github.io/)

## Introduction

In real-world scenarios, the model should handle a dynamic shift of audiovisual data distribution when training on videos, as the agent's surroundings can continuously change over time.

We propose STELLA (Spatio-Temporal Localized Alignment), which captures the complex relationships between the audio and video modalities during training on a sequence of pre-training tasks while alleviating the forgetting of learned audiovisual correlations.

<center><img src="assets/concept_figure_16x9.png" alt="Teaser" width="100%"></center>
<p>
Our method harnesses cross-modal attention maps from the AVM module to compute importance scores in order to identify highly correlated patches (<b>Localized Patch Importance Scoring</b>).
        Comparing the attention maps created by the current queries with those generated by past queries, we compute correlation scores of the current patches with the past data (<b>Replay-guided Correlation Assessment</b>).
        Finally, we perform a probabilistic patch selection, combining the importance scores and correlation scores to select patches for continual audio-video pre-training (<b>Multimodal Patch Selection for Continual Learning</b>).
</p>

## Install
### Setup `python` environment
```
conda create -n stella python=3.7
```

### Install `pytorch`, `torchvision`, and `torchaudio`
The following version have been tested.  
* `torch  1.9.1`
* `torchvision  0.14.1` 
* `torchaudio  0.9.1`  

You can try other version of `pytorch` but make sure that it will be compatible with your `cuda` and `cudnn`.  

### Install other dependencies
```
pip install -r requirements.txt
```

## Dataset
We refer [CAV](https://github.com/YuanGongND/cav-mae) to consist datasets for our experiments. Download .json files of the VGGSound and AudioSet datasets.<br>
Place the downloaded files in /dataset_path/vggsound/data and /dataset_path/AudioSet/data, respectively.

### VGGSound

1. Follow the instruction in [audiosetdl](https://github.com/speedyseal/audiosetdl) to download the VGGSound dataset.
2. Move current directory to /tools/vggsound/data_preprocess
3. Run ```python add_category_info_vggsound.py``` to add category information.
4. Run ```python json2csv_vggsound.py``` to change .json files into .csv files.
5. Run ```python split_vggsound.py``` to split the dataset into subset datasets.
6. Run ```python valid_clip_vggsound.py``` to find valid clips of the VGGSound.
7. Run ```python extract_audio.py``` & ```python extract_video_freame.py``` to extract audio & video from the clips.

### AudioSet

1. Follow the instruction in [audioset-processing](https://github.com/aoifemcdonagh/audioset-processing) to download all the required class datasets in the AudioSet dataset.<br>
   When downloading the samples, <i>exclude samples that are included in more than two different categories</i>.
2. Move current directory to /tools/vggsound/data_preprocess
3. Run ```python retrieval_task_audioset.py``` to consist .csv file for the AudioSet retrieval task.
3. Follow the instruction in [audiosetdl](https://github.com/speedyseal/audiosetdl) and the .csv file in 2. to download samples for the retrieval task.
4. Run ```python add_category_info_audioset.py``` to add category information.
5. Run ```python split_audioset.py``` to split the dataset into subset datasets.
6. Run ```python cl_train_test_audioset.py``` to follow the same format of the train/test VGGSound .csv files.
7. Run ```python cl_retrieval_audioset.py``` to follow the same format of the retrieval VGGSound .csv file.
8. Run ```python extract_audio.py``` & ```python extract_video_freame.py``` to extract audio & video from the clips.

### MST-VTT

1. Download the dataset through "https://www.kaggle.com/datasets/vishnutheepb/msrvtt".
2. Move current directory to /tools/msrvtt
3. Run ```python extract_audio.py``` & ```python extract_video_freame.py``` to extract audio & video from the clips.
4. Run ```python find_valid_audio_msrvtt.py``` to delete files that does not contain audio modality.

### AVE

1. Download the datset through "https://sites.google.com/view/audiovisualresearch"
2. Run ```python extract_audio.py``` & ```python extract_video_freame.py``` to extract audio & video from the clips.


### AVS

1. Download the datset through "https://github.com/OpenNLPLab/AVSBench"
2. Run ```python extract_audio.py``` & ```python extract_video_freame.py``` to extract audio & video from the clips.


## Training

### Initial weights
```
python tools/adapt_imae_weights_to_vamae.py
bash commands/cav_pretrain_audioset20k_base_init_check.sh
bash commands/cav_pretrain_submodule_audioset20k_base_init_check.sh
bash commands/cav_pretrain_vggsound_base_init_check.sh
bash commands/cav_pretrain_submodule_vggsound_base_init_check.sh
```

### Continual Pretraining
```
bash commands/cav_pretrain_{dataset}_base.sh
```

### Downstream task (Retrieval)

```
bash commands/cav_finetune_{downstreamtask}_base.sh
```

## Bibtex
```
@article{lee2023lifelong,
      title={STELLA: Continual Audio-Video Pre-training with Spatio-Temporal Localized Alignment},
      author={Jaewoo Lee and Jaehong Yoon and Wonjae Kim and Yunji Kim and Sung Ju Hwang},
      journal={arXiv preprint arXiv:2310.08204},
      year={2024},
      url={https://doi.org/10.48550/arXiv.2310.08204},
}
```
