set -Eeuo pipefail
trap 'rc=$?; echo "Error: \"$BASH_COMMAND\" failed at $BASH_SOURCE:$LINENO (exit $rc)"; exit $rc' ERR

service ssh start

set -euo pipefail

python_output="$(python3 steering_reasoning/helpers/check_training_setup.py | head -n 1 | tr -d '\r\n' )"

if [[ "$python_output" == "steering" || "$python_output" == "steering_rank" ]]; then
    echo "It is steering. Modifying the files"
    bash bin/helpers/modify_bias.sh
fi


if [ -n "${ADD_PLACE:-}" ]; then
    model_dir="$(d=$(find /from_s3/models -mindepth 1 -maxdepth 1 -type d -print); [ "$(printf '%s\n' "$d" | wc -l)" -eq 1 ] && printf '%s' "$d" || { echo 'Expected exactly one subdir in /from_s3/models' >&2; exit 1; })"

    python3 -m steering_reasoning.helpers.modify_config \
        --model_dir $model_dir \
        --cfg_path config.json \
        --add_place $ADD_PLACE
fi
