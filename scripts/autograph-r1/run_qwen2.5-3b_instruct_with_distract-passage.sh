# run on 8xH20
# make sure your current working directory is the root of the project
# export CUDA devices
#!/bin/bash
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
# export RAY_BACKEND_LOG_LEVEL=debug
# export RAY_LOG_TO_STDERR=1
# Function to get a value from the config file
# max_assistant_turn = max_user_turn * 2 
# max_user_turn - 1 = number of tool_call
CONFIG_FILE="verl/third_party/autograph_r1/config.ini"
get_config_value() {
    local section=$1
    local key=$2
    awk -F '=' -v section="[$section]" -v key="$key" '
    $0 ~ section { in_section=1; next }
    /^\[.*\]/ { in_section=0 }
    in_section && $1 ~ key { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }
    ' "$CONFIG_FILE"
}

# Example: Get values from the config file
WANDB_API_KEY=$(get_config_value "logging" "WANDB_API_KEY")
# Print the values (for debugging)
export WANDB_API_KEY
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"

# AutoGraph parameters
DIFFFICULTY="medium" # available: easy, medium
DOC_SIZE=15 # available: 8,12,15
WITH_DISTRACT="True" # Only True is supported now
TEXT_LINKING="False" # available: True, False

 /data/autograph/data/musique_validation_doc_size_15_distract_False_with_mcq_False_difficulty_medium_text_linking_False.parquet
TRAIN_DATA="/data/autograph/data/musique_train_doc_size_${DOC_SIZE}_distract_${WITH_DISTRACT}_with_mcq_False_difficulty_${DIFFFICULTY}_text_linking_${TEXT_LINKING}.parquet"
VAL_DATA="/data/autograph/data/musique_validation_doc_size_${DOC_SIZE}_distract_${WITH_DISTRACT}_with_mcq_False_difficulty_${DIFFFICULTY}_text_linking_${TEXT_LINKING}.parquet"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

CHECKPOINT_DIR="/data/autograph/checkpoints/${TIMESTAMP}_qwen2.5-7b-autograph-distract_${DIFFFICULTY}-docsize${DOC_SIZE}-textlinking${TEXT_LINKING}"

if [ "$TEXT_LINKING" = "True" ]; then
    reward_fn_file_path="verl/third_party/autograph_r1/precision_reward.py"
else
    reward_fn_file_path="verl/third_party/autograph_r1/reward.py"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='autograph_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path='config/interaction_config/autograph_interaction_config.yaml' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='auto_graph_rl' \
    trainer.experiment_name="azure-qwen2.5-3b-auto-graph-rl-distract-${DIFFFICULTY}-docsize${DOC_SIZE}-text-linking${TEXT_LINKING}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=50 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.ray_wait_register_center_timeout=3600 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    custom_reward_function.path="$reward_fn_file_path" \
    