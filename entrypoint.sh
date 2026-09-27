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
MODELS=${OLLAMA_MODELS:-${MODEL_NAME:-"qwen2.5:7b-instruct-q4_K_M"}}
for model in $MODELS; do
    echo "🔴 Ensuring model $model is available..."
    ollama pull "$model"
done

# Dynamic Modelfile generation and fenced model build
RENDERED_MODELFILE="/tmp/Modelfile.rendered"
TEMPLATE_PATH=""

if [ -f "/scripts/Modelfile.template" ]; then
    TEMPLATE_PATH="/scripts/Modelfile.template"
elif [ -f "/app/scripts/Modelfile.template" ]; then
    TEMPLATE_PATH="/app/scripts/Modelfile.template"
fi

BASE_MODEL=${OLLAMA_BASE_MODEL:-"qwen2.5:7b-instruct-q4_K_M"}
NUM_CTX=${OLLAMA_NUM_CTX:-4096}
TEMP=${OLLAMA_TEMPERATURE:-0.2}

if [ -n "$TEMPLATE_PATH" ]; then
    echo "🛡️ Rendering dynamic Modelfile from $TEMPLATE_PATH (num_ctx: $NUM_CTX, temp: $TEMP)..."
    sed -e "s|\${OLLAMA_BASE_MODEL:-[^}]*}|$BASE_MODEL|g" \
        -e "s|\${OLLAMA_NUM_CTX:-[^}]*}|$NUM_CTX|g" \
        -e "s|\${OLLAMA_TEMPERATURE:-[^}]*}|$TEMP|g" \
        "$TEMPLATE_PATH" > "$RENDERED_MODELFILE"
    ollama create qwen2.5:7b-fenced -f "$RENDERED_MODELFILE"
elif [ -f "/scripts/Modelfile" ]; then
    echo "🛡️ Building fenced model from static /scripts/Modelfile..."
    ollama create qwen2.5:7b-fenced -f "/scripts/Modelfile"
elif [ -f "/root/Modelfile" ]; then
    echo "🛡️ Building fenced model from static /root/Modelfile..."
    ollama create qwen2.5:7b-fenced -f "/root/Modelfile"
fi


echo "🟢 Done!"

# Keep the process running
wait $pid
