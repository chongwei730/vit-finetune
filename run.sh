#!/bin/bash
MODEL_NAME=swin
for i in {0..3}; do
  CONFIG_PATH=$(sed -n "$((i+1))p" ${MODEL_NAME}_config_list.txt) 
  SCHEDULER_NAME=$(echo "$CONFIG_PATH" | cut -d'/' -f2)

  echo "Launching on GPU $i:"
  echo "  CONFIG: $CONFIG_PATH"
  echo "  SCHEDULER: $SCHEDULER_NAME"


  CUDA_VISIBLE_DEVICES=$i python main.py fit \
    --config "$CONFIG_PATH" \
    --trainer.logger.name="$SCHEDULER_NAME/$MODEL_NAME" > ./log/log_gpu${i}.out 2>&1 &
done


wait
echo "All 4 tasks completed."