#!/bin/bash

# Start Ollama in the background
ollama serve &
pid=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 2
done

# Pull the model if not already present
echo "🔴 Ensuring model deepseek-r1:1.5b is available..."
ollama pull deepseek-r1:1.5b
echo "🟢 Done!"

# Keep the process running
wait $pid
