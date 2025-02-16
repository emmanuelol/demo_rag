# demo_rag

This repository is a demo of building and using a RAG application based on LangChain, Ollama, and DeeepSeek. There are two approaches to this task: one is based on running locally, and the second is how to run in the Google Cloud Platform.

Local
The local application is based on Docker Compose and NVIDIA container tools. So please consider that as requirements, the files follow the next order:

demo_rag/
├─ client/
├─ requirements.txt
├─ Dockerfile
docs/
LICENSE
README.md
docker-compose.yaml
entrypoint.sh
entrypoint_rag.sh/
run_rag.py

To execute it, access the demo_rag directory, use the following commands in a terminal, and open the URL that Gradio will provide.
```
docker-compose up
```
To close all, as above, access the demo_rag directory and use the command:
```
docker-compose down
```

![UI for RAG](https://github.com/emmanuelol/demo_rag/blob/main/docs/Captura%20desde%202025-02-14%2018-36-35.png) 
## Collaborators
Emmanuel Ortiz Lopez ([emmanuelol](https://github.com/emmanuelol)) :octocat:
Carlos Armando Ortiz Lopez ([carlosaol](https://github.com/carlosaol))