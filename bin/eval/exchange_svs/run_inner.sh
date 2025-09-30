
trap 'rc=$?; echo "Error: \"$BASH_COMMAND\" failed at $BASH_SOURCE:$LINENO (exit $rc)"; exit $rc' ERR
bash bin/helpers/modify_bias.sh

poetry run python3 -m steering_reasoning.helpers.exchange_svs --config_path configs/helpers/exchange_svs.yml

bash bin/eval/vanilla/run_inner.sh