#!/bin/bash
## get path
full_path=$(readlink -f $0)
dir_path=$(dirname $full_path)

# create network
docker network create ollama-net

# Default values
## the current script is using a production image, to use for development please provide as argument for this script -i aidex_dev
image_name=ollama/ollama
#This paths are set for the ZUD0066u server if you want to use on other device please update the paths to your setup
## bdd and model path
datasets_path='/media/Datos/datasets/'
models_path='/media/Datos/models/'
## change the name of your container
container_name=client_ollama
## if needed please change the port
port=8004

### get the paths
while getopts b:m:d:e:c:i:p: flag
do
    case "${flag}" in
        m) models_path=${OPTARG};;
        d) datasets_path=${OPTARG};;
        c) container_name=${OPTARG};;
        p) port=${OPTARG};;
        i) image_name==${OPTARG};;
    esac
done


#docker run --name $container_name -d -it --rm --privileged \
#--network=host --gpus all --shm-size 16G \
#-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
#-e OLLAMA_HOST=127.0.0.1:11434 \
#-v /tmp/.X11-unix:/tmp/.X11-unix  \
#-v $datasets_path:/datasets \
#-v $models_path:/models \
#-v $dir_path:/app $image_name 

docker run --name $container_name -d --rm --privileged \
--network=host --gpus all --shm-size 16G \
-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
-e OLLAMA_HOST="0.0.0.0" \
-e OLLAMA_NUM_PARALLEL=1 \
-e OLLAMA_MAX_LOADED_MODELS=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-v $datasets_path:/datasets \
-v $models_path:/models \
-v $dir_path:/app $image_name

# Wait for Ollama to become responsive
echo "Waiting for Ollama container to be responsive..."
docker exec $container_name bash -c "until curl -s http://127.0.0.1:11434 > /dev/null; do sleep 2; done"

# Build and verify fenced model with parameterized context and temperature
echo "Configuring fenced model qwen2.5:7b-fenced..."
docker exec $container_name ollama pull ${OLLAMA_BASE_MODEL:-qwen2.5:7b-instruct-q4_K_M}

docker exec $container_name bash -c '
    NUM_CTX=${OLLAMA_NUM_CTX:-4096}
    TEMP=${OLLAMA_TEMPERATURE:-0.2}
    BASE_MODEL=${OLLAMA_BASE_MODEL:-"qwen2.5:7b-instruct-q4_K_M"}
    if [ -f "/app/Modelfile.template" ]; then
        sed -e "s|\${OLLAMA_BASE_MODEL:-[^}]*}|$BASE_MODEL|g" \
            -e "s|\${OLLAMA_NUM_CTX:-[^}]*}|$NUM_CTX|g" \
            -e "s|\${OLLAMA_TEMPERATURE:-[^}]*}|$TEMP|g" \
            /app/Modelfile.template > /tmp/Modelfile.rendered
        ollama create qwen2.5:7b-fenced -f /tmp/Modelfile.rendered
    else
        ollama create qwen2.5:7b-fenced -f /app/Modelfile
    fi
'
echo "🟢 Fenced model qwen2.5:7b-fenced ready."

docker exec -it $container_name bash