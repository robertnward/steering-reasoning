bash bin/helpers/train_bash_commands.sh

if [ -n "${ADD_PLACE}" ]; then
    export PYTHONPATH=/workspace:$PYTHONPATH
    export VLLM_NO_USAGE_STATS=1
    sed -i 's/if weights_not_loaded/if False and weights_not_loaded/g' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/model_loader/loader.py
fi

# Continuously check until MASTER_ADDR is resolved
while true; do
    if ! output=$(nslookup "$MASTER_ADDR" 2>/dev/null); then
        # nslookup failed → force MASTER_IP to empty
        MASTER_IP=""
    else
        # nslookup succeeded → extract the first “clean” Address line
        MASTER_IP=$(
            printf '%s\n' "$output" |
            awk '/^Address: / && $2 !~ /#/ { print $2; exit }'
        )
    fi
    echo $MASTER_IP
    if [ -n "$MASTER_IP" ]; then
        echo "MASTER_ADDR resolved to: $MASTER_IP"
        break
    else
        echo "Waiting for $MASTER_ADDR to be resolved..."
        sleep 5
    fi
done

echo "Resolved MASTER_IP: $MASTER_IP"

ray start --address="${MASTER_IP:=0}:6379" 
sleep infinity