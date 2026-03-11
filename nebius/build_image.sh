#!/bin/bash
set -e

# Parse command line arguments
PUSH_FLAG=false
LOGIN_FLAG=false
dynamo_version=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH_FLAG=true
            shift
            ;;
        --login)
            LOGIN_FLAG=true
            shift
            ;;
        --dynamo)
            dynamo_version="$2"
            if [ -z "$dynamo_version" ]; then
                echo "Error: --dynamo requires a version (e.g. --dynamo 0.9.1)"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set base image
image="nvcr.io/nvidia/tensorrt-llm/release"

# Read version from version.py
ver=$(grep '__version__' version.py | cut -d'"' -f2)

# Split version into trtllm_ver and nb_ver by '+'
if [[ "$ver" == *"+"* ]]; then
    trtllm_ver="${ver%+*}"
    nb_ver="${ver#*+}"
else
    echo "Error: Version format should be 'trtllm_ver+nb_ver', got: $ver"
    exit 1
fi

# check that nebius version matches
ver_trt=$(grep '__version__' ../tensorrt_llm/version.py | cut -d'"' -f2)

if [ "$ver_trt" != "$trtllm_ver" ]; then
    echo "Error: TensorRT-LLM version mismatch: nebius/version.py has trtllm_ver=$trtllm_ver but tensorrt_llm/version.py has __version__=$ver_trt"
    exit 1
fi

commit_hash=$(git rev-parse --short HEAD)

# Check for uncommitted changes in tensorrt_llm directory
uncommitted=$(git status --porcelain ../tensorrt_llm/)
if [ -n "$uncommitted" ]; then
    echo "Warning: Uncommitted changes in tensorrt_llm directory (will be ignored):"
    echo "$uncommitted" | sed 's/^/  /'
    echo "These changes will not be included in the patch."
    echo
    read -p "continue (y/n)? " reply
    if [[ ! "$reply" =~ ^[yY]$ ]]; then
        exit 1
    fi
fi

echo "TensorRT-LLM version: $trtllm_ver"
echo "Nebius version: $nb_ver"
[ -n "$dynamo_version" ] && echo "Dynamo version: $dynamo_version"
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
if [ -n "$dynamo_version" ]; then
    image_tag="$trtllm_ver.$nb_ver.dynamo$dynamo_version.$commit_hash"
else
    image_tag="$trtllm_ver.$nb_ver.$commit_hash"
fi
built_image="$image:nb.dev"

# Use BASE_IMAGE env var if set, otherwise use dynamo or default
if [ -n "$BASE_IMAGE" ]; then
    base_image="$BASE_IMAGE"
elif [ -n "$dynamo_version" ]; then
    base_image="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:$dynamo_version"
else
    base_image="$image:$trtllm_ver"
fi

echo "Building Docker image with BASE_IMAGE=$base_image"
docker build -t "$built_image" . --build-arg BASE_IMAGE="$base_image" --platform=linux/amd64

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

# Check if images.txt exists
IMAGE_LIST_FILE="images.txt"
if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Error: File $IMAGE_LIST_FILE does not exist"
    exit 1
fi

# Prompt for credentials when --login and --push are both set
if [ "$PUSH_FLAG" = true ] && [ "$LOGIN_FLAG" = true ]; then
    read -p "user (iam): " DOCKER_USER
    DOCKER_USER="${DOCKER_USER:-iam}"
    read -s -p "password: " DOCKER_PASSWORD
    echo
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
        if [ "$LOGIN_FLAG" = true ] && [ -n "$DOCKER_PASSWORD" ]; then
            registry="${result_image%%/*}"
            echo "Logging in to registry: $registry"
            echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USER" --password-stdin "$registry"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to login to registry $registry"
                continue
            fi
        fi
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
