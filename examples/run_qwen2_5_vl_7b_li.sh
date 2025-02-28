set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=2,3

MODEL_PATH=/lpai/models/qwen2-5-vl-7b-instruct/25-02-18-1  # replace it with your local file path

python3 -m verl.trainer.main \
    config=grpo_example.yaml \
    data.train_files=/lpai/EasyR1/data/train \
    data.val_files=/lpai/EasyR1/data/test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=0.05 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=qwen2_5_vl_7b_geo \
    trainer.n_gpus_per_node=2 \
    data.max_prompt_length=4096 \
    data.max_pixels=655366  \
    
    
