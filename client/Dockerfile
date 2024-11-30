FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace

COPY src /workspace/src

RUN pip install fastapi uvicorn requests httpx xlsxwriter

CMD ["tail", "-f", "/dev/null"]