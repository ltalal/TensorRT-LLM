#!/bin/bash
set -e

# Parse command line arguments
PUSH_FLAG=false
if [[ "$1" == "--push" ]]; then
    PUSH_FLAG=true
fi

# Set base image
image="nvcr.io/nvidia/tensorrt-llm/release"

# Read version from version.py
ver=$(grep '__version__' ../tensorrt_llm/version.py | cut -d'"' -f2)

# Split version into trtllm_ver and nb_ver by '+'
if [[ "$ver" == *"+"* ]]; then
    trtllm_ver="${ver%+*}"
    nb_ver="${ver#*+}"
else
    echo "Error: Version format should be 'trtllm_ver+nb_ver', got: $ver"
    exit 1
fi

commit_hash=$(git rev-parse --short HEAD)

echo "TensorRT-LLM version: $trtllm_ver"
echo "Nebius version: $nb_ver"
echo "Commit hash: $commit_hash"

# Make patch from current branch to tag "v$trtllm_ver", limited to tensorrt_llm directory
echo "Creating patch from current branch to tag v$trtllm_ver for tensorrt_llm directory..."
git diff "v$trtllm_ver"..HEAD -- ../tensorrt_llm/ > patch.txt

if [ ! -s patch.txt ]; then
    echo "Warning: No differences found between current branch and v$trtllm_ver"
fi

echo "Applying patch:"
cat patch.txt

# Build Docker image
image_tag="$trtllm_ver.$nb_ver.$commit_hash"
built_image="$image:$image_tag"
echo "Building Docker image with BASE_IMAGE=$image:$trtllm_ver"
docker build -t "$built_image" . --build-arg BASE_IMAGE="$image:$trtllm_ver" --platform=linux/amd64

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

export TRTLLM_IMAGE=$built_image

# Check if images.txt exists
IMAGE_LIST_FILE="images.txt"
if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Error: File $IMAGE_LIST_FILE does not exist"
    exit 1
fi

# Read images.txt and tag the resulting image
while IFS= read -r result_image || [[ -n "$result_image" ]]; do
    # Skip empty lines and comments
    if [[ -z "$result_image" || "$result_image" =~ ^# ]]; then
        continue
    fi

    target_tag="$result_image:$image_tag"
    echo "Tagging image as: $target_tag"
    docker tag "$built_image" "$target_tag"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to tag image as $target_tag"
        continue
    fi

    if [ "$PUSH_FLAG" = true ]; then
        echo "Pushing image: $target_tag"
        docker push "$target_tag"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to push image $target_tag"
        else
            echo "Successfully pushed $target_tag"
        fi
    else
        echo "Image tagged as $target_tag (not pushing, use --push to push)"
    fi

    echo "-----------------------------------------------------"
done < "$IMAGE_LIST_FILE"

echo "All images processed!"
