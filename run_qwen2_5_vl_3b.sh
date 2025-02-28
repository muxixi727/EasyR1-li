set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=/lpai/models/qwen2-5-vl-3b-instruct/25-02-18-1  # replace it with your local file path

python3 -m verl.trainer.main \
    config=example/grpo_example.yaml \
    data.train_files=/lpai/EasyR1/data/train \
    data.val_files=/lpai/EasyR1/data/test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.8 \
    trainer.experiment_name=qwen2_5_vl_3b_geo \
    trainer.n_gpus_per_node=8 \
    data.max_prompt_length=4096 \
    data.max_pixels=655366