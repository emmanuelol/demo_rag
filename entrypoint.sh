#!/bin/bash

# Start Ollama in the background
ollama serve &
pid=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 2
done

# Pull the model(s) if not already present
MODELS=${OLLAMA_MODELS:-${MODEL_NAME:-"deepseek-r1:1.5b"}}
for model in $MODELS; do
    echo "🔴 Ensuring model $model is available..."
    ollama pull "$model"
done
echo "🟢 Done!"

# Keep the process running
wait $pid
