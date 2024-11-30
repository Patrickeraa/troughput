FROM nvcr.io/nvidia/pytorch:23.05-py3

# 
WORKDIR /workspace

RUN pip install fastapi uvicorn torchvision torch xlsxwriter pandas

# 
COPY main.py /workspace/server/

# "fastapi", "run", "server/main.py", "--host", "0.0.0.0", "--port", "8050"
CMD ["tail", "-f", "/dev/null"]