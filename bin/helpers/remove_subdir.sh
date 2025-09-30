if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <parent-directory>" >&2
  exit 1
fi

PARENT=$1

subdirs=( "$PARENT"/*/ )

# Sanity check - we expect exactly one subdirectory
if (( ${#subdirs[@]} != 1 )); then
  echo "Error: expected exactly one subdirectory in '$PARENT', found ${#subdirs[@]}." >&2
  exit 1
fi

SUBDIR=${subdirs[0]%/}
echo $SUBDIR

mv -- "$SUBDIR"/* "$PARENT"/

# Remove the now-empty subdirectory
rmdir -- "$SUBDIR"