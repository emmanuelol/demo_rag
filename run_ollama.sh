#!/bin/bash
## get path
full_path=$(readlink -f $0)
dir_path=$(dirname $full_path)


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


docker run --name $container_name -d -it --rm --privileged \
--network=host --gpus all --shm-size 16G \
-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
-e OLLAMA_HOST=127.0.0.1:11434 \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-v $datasets_path:/datasets \
-v $models_path:/models \
-v $dir_path:/app $image_name 


#VAR1='jupyter lab --allow-root --no-browser --port='
#cmd="${VAR1}${port}"
cmd = 'export OLLAMA_HOST=127.0.0.1:11435'
docker exec -it $container_name bash -c "$cmd"
docker exec -it $container_name bash