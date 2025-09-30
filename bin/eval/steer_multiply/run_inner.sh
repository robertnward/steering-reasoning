bash bin/helpers/modify_bias.sh

python3 -m steering_reasoning.helpers.insert_vector --steer_multiply $STEER_MULTIPLY

bash bin/eval/vanilla/run_inner.sh