#!/bin/bash

# Function to check if Ollama is ready
wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    until curl -s http://ollama:11434 > /dev/null; do
        sleep 2
    done
    echo "Ollama is ready!"
}

# Wait for Ollama service
wait_for_ollama

# Start Jupyter Lab in the background if JUPYTER_PORT is set
if [ ! -z "$JUPYTER_PORT" ]; then
    echo "Starting Jupyter Lab on port $JUPYTER_PORT..."
    jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=$JUPYTER_PORT --NotebookApp.token='' --NotebookApp.password='' &
fi

# Start the RAG application
python /app/run_rag.py
