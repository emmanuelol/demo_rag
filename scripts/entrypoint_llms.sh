#!/bin/bash
#export OLLAMA_HOST=127.0.0.1:11435

#export OLLAMA_HOST=locahost:11435

#export OLLAMA_HOST=0.0.0.0:11435

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve  models..."
#ollama pull all-minilm:l6
#ollama pull mistral
ollama pull bge-m3
#ollama pull llama3.1:8b
ollama pull deepseek-r1:1.5b
echo "🟢 Done!"
sleep 5
# Wait for Ollama process to finish.

wait $pid
ollama serve
