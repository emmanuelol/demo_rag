#!/bin/bash
## get path
full_path=$(readlink -f $0)
dir_path=$(dirname $full_path)

# Default values
image_name=client_python
datasets_path='/mnt/d/datasets/'
models_path='/mnt/d/models/'
container_name=client_python
port=8003

### get the paths
while getopts b:m:d:e:c:i:p: flag
do
    case "${flag}" in
        m) models_path=${OPTARG};;
        d) datasets_path=${OPTARG};;
        c) container_name=${OPTARG};;
        p) port=${OPTARG};;
        i) image_name=${OPTARG};;
    esac
done

# Stop existing container
docker stop $container_name 2>/dev/null || true

# Use port mapping instead of --network=host
docker run --name $container_name -d -it --rm --privileged \
--network=ollama-net -p $port:$port --gpus all --shm-size 16G \
-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-v $datasets_path:/datasets \
-v $models_path:/models \
-v $dir_path:/app $image_name bash

# Start Jupyter Lab
echo "Starting Jupyter Lab on port $port..."
docker exec -d $container_name bash -c "cd /app && jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=$port --NotebookApp.token='' --NotebookApp.password=''"

# Wait for startup
sleep 5

echo ""
echo "Jupyter Lab should be accessible at:"
echo "From WSL2: http://localhost:${port}/lab"
echo "From Windows: http://localhost:${port}/lab"
echo ""

# Test connection
echo "Testing connection..."
curl -s http://localhost:$port > /dev/null && echo "✅ Connection successful!" || echo "❌ Connection failed"

# Attach to container
docker exec -it $container_name bash