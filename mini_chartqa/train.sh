#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4,5,6
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=datasets/train_full.parquet \
    data.val_files=datasets/val_full.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=2 \
    trainer.experiment_name=mini_chartQA \
    trainer.n_gpus_per_node=4 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    data.rollout_batch_size=16 \
    data.val_batch_size=256 \
    trainer.save_checkpoint_path=./checkpoints/mini_chartQA \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
