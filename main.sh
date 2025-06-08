#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --account=jusun
#SBATCH --gres=gpu:1
#SBATCH --output=exp_log/exp_%a.out
#SBATCH --error=exp_log/exp_%a.err
#SBATCH --job-name=scheduler

#SBATCH -p v100


conda activate finetune

CONFIG_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" config_list.txt)


SCHEDULER_NAME=$(echo "$CONFIG_PATH" | cut -d'/' -f2)

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running config file: $CONFIG_PATH"
echo "Using scheduler: $SCHEDULER_NAME"


python main.py fit \
    --config "$CONFIG_PATH" \
    --trainer.logger.name="$SCHEDULER_NAME"