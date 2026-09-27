#!/bin/bash

# Function to check if Ollama is ready
wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    until curl -s http://${BASE_URL#http://} > /dev/null 2>&1 || curl -s http://ollama:11434 > /dev/null 2>&1; do
        sleep 2
    done
    echo "Ollama is ready!"
}

# Function to check if Qdrant is ready
wait_for_qdrant() {
    local q_host=${QDRANT_HOST:-qdrant}
    local q_port=${QDRANT_PORT:-6333}
    echo "Waiting for Qdrant at ${q_host}:${q_port} to be ready..."
    until curl -sf "http://${q_host}:${q_port}/readyz" > /dev/null 2>&1; do
        sleep 2
    done
    echo "Qdrant is ready!"
}

# Volume pre-flight permission check
for dir in "/datasets" "/models" "/tmp"; do
    if [ -d "$dir" ] && [ ! -w "$dir" ]; then
        echo "⚠️ Warning: Volume $dir is not writable by $(id -u):$(id -g)."
    fi
done

# Wait for backend dependencies
wait_for_ollama
wait_for_qdrant


# Start Jupyter Lab in the background if JUPYTER_PORT is set
if [ ! -z "$JUPYTER_PORT" ]; then
    echo "Starting Jupyter Lab on port $JUPYTER_PORT..."
    jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=$JUPYTER_PORT --NotebookApp.token='' --NotebookApp.password='' &
fi

# Start the RAG application
python /app/run_rag.py
