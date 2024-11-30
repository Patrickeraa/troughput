from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from queue import Queue
import threading
import time
import pandas as pd
from llama_cpp import Llama
import asyncio

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    prompt: str
    response: str
    processing_time: float
    generated_tokens: int
    finish_reason: str

gpu_layers = -1
ctx = 2048
max_tk = 5000
seed = 42
rep_penalty = 1.1

llm = Llama(
    model_path="/workspace/dataset/l7b/ggml-model-f16.gguf",
    n_gpu_layers=gpu_layers,
    n_ctx=ctx
)

request_queue = Queue()
results = {}

def process_queue():
    while True:
        if not request_queue.empty():
            request_id, prompt = request_queue.get()
            start_time = time.time()

            output = llm(
                prompt,
                max_tokens=max_tk,
                echo=False,
                stop=["Q:", "A:"],
                seed=seed,
                repeat_penalty=rep_penalty
            )
            end_time = time.time()

            results[request_id] = {
                "response": output['choices'][0]['text'],
                "generated_tokens": output['usage']['completion_tokens'],
                "processing_time": end_time - start_time,
                "finish_reason": output['choices'][0].get('finish_reason', 'unknown')
            }
            request_queue.task_done()

worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

@app.post("/process-prompt/", response_model=PromptResponse)
async def process_prompt(request: PromptRequest, background_tasks: BackgroundTasks):
    request_id = f"{time.time_ns()}"
    request_queue.put((request_id, request.prompt))
    while request_id not in results:
        await asyncio.sleep(0.1)
    result = results.pop(request_id)
    return PromptResponse(
        prompt=request.prompt,
        response=result["response"],
        processing_time=result["processing_time"],
        generated_tokens=result["generated_tokens"],
        finish_reason=result["finish_reason"]
    )