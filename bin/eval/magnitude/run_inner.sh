bash bin/helpers/modify_bias.sh

python3 -m steering_reasoning.helpers.steer_multiply --factor $FACTOR

bash bin/eval/vanilla/run_inner.sh