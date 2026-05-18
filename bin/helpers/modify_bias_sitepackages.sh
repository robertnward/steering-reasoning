#!/bin/bash
# Site-packages-aware version of the upstream bin/helpers/modify_bias_vllm.sh.
# Upstream hardcodes /usr/local/lib/python3.11/dist-packages; we accept a path arg.
#
# Usage: bash modify-bias-sitepackages.sh /path/to/vllm/install
# Or:    bash modify-bias-sitepackages.sh "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
#
# What it does: enables a bias term on the MLP down_proj RowParallelLinear for
# qwen2 / llama / olmo / olmo2 vllm model classes. This is the runtime mechanism
# by which Sinii's steering vector is added as an additive bias.

set -euo pipefail

VLLM_DIR="${1:?usage: $0 <path-to-vllm-install>}"

if [ ! -d "$VLLM_DIR/model_executor/models" ]; then
  echo "Expected $VLLM_DIR/model_executor/models to exist. Is this a vllm install dir?" >&2
  exit 1
fi

# allow loading partial weights when we add a bias that wasn't there
LOADER="$VLLM_DIR/model_executor/model_loader/loader.py"
if [ -f "$LOADER" ]; then
  sed -i 's/if weights_not_loaded/if False and weights_not_loaded/g' "$LOADER"
fi

# enable bias on down_proj for each model class
patch_bias() {
  local f="$1"
  local pattern="$2"
  if [ -f "$f" ]; then
    sed -E -i "
      /^\s*self\.down_proj\s*=\s*RowParallelLinear\s*\(/,/^\s*\)\s*$/{
          s/$pattern/bias=True/
      }
    " "$f"
    echo "patched $f"
  else
    echo "skipped (not present): $f"
  fi
}

patch_bias "$VLLM_DIR/model_executor/models/qwen2.py"  '\bbias=False\b'
patch_bias "$VLLM_DIR/model_executor/models/llama.py"  '\bbias=bias\b'
patch_bias "$VLLM_DIR/model_executor/models/olmo2.py"  '\bbias=False\b'
patch_bias "$VLLM_DIR/model_executor/models/olmo.py"   '\bbias=False\b'

echo "vllm bias patches applied at $VLLM_DIR"
