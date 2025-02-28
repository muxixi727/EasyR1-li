set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=2,3

MODEL_PATH=/lpai/models/qwen2-5-vl-3b-instruct/25-02-18-1  # replace it with your local file path

python3 -m verl.trainer.main \
    config=grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=qwen2_5_vl_3b_geo \
    trainer.n_gpus_per_node=2 \
    data.max_prompt_length=4096 \
    data.rollout_batch_size=512
