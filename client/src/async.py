import requests
import time
import threading
import pandas as pd
from statistics import mean, median
import numpy as np

SERVER_URL = "http://server:8050/process-prompt/"

results = []
resultsxlsx = []
lock = threading.Lock()

def send_prompt(prompt: str, prompt_number: int):

    submission_time = time.time() 
    response = requests.post(SERVER_URL, json={"prompt": prompt})
    reception_time = time.time()

    if response.status_code == 200:
        result = response.json()
        total_time = reception_time - submission_time
        processing_time = result["processing_time"]
        waiting_time = total_time - processing_time

        generated_tokens = result["generated_tokens"]
        tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
        finish_reason = result["finish_reason"]

        with lock:
            results.append({
                "Prompt Number": prompt_number,
                "Prompt": prompt,
                "Generated Text": result["response"],
                "Total Time (s)": total_time,
                "Waiting Time (s)": waiting_time,
                "Processing Time (s)": processing_time,
                "Tokens per Second": tokens_per_second,
                "Generated Tokens": generated_tokens,
                "Finish Reason": finish_reason
            })
            resultsxlsx.append({
                "Prompt Number": prompt_number,
                "Total Time (s)": total_time,
                "Waiting Time (s)": waiting_time,
                "Processing Time (s)": processing_time,
                "Tokens per Second": tokens_per_second,
                "Generated Tokens": generated_tokens,
                "Finish Reason": finish_reason
            })
    else:
        print(f"Error: Server returned status code {response.status_code} for prompt {prompt_number}")

if __name__ == "__main__":
    prompts = [
        "Talk about our sun",
        "Talk about the planet Mercury",
        "Talk about the planet Venus",
        "Talk about the planet Earth",
        "Talk about the planet Mars",
        "Talk about the planet Jupyter",
        "Talk about the planet Saturn",
        "Talk about the planet Uranus",
        "Talk about the planet Neptune",
        "Talk about the moons of Jupyter",
        "Talk about the moons of Saturn",
        "Talk about the moons of Uranus",
        "Talk about the moons of Neptune",
        "Talk about the closes solar system from ours",
        "Talk about the Milky Way",
        "Talk about the Andromeda galaxy"
    ]

    print("Sending prompts to the Llama model with multithreading...\n")
    threads = []

    # Start a thread for each prompt
    for prompt_number, prompt in enumerate(prompts, start=1):
        thread = threading.Thread(target=send_prompt, args=(prompt, prompt_number))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Compute Metrics
    total_times = [r["Total Time (s)"] for r in results]
    waiting_times = [r["Waiting Time (s)"] for r in results]
    processing_times = [r["Processing Time (s)"] for r in results]
    tokens_per_second_list = [r["Tokens per Second"] for r in results]

    metrics = {
        "Total Time Mean": mean(total_times),
        "Total Time Median": median(total_times),
        "Total Time 99th Percentile": np.percentile(total_times, 99),
        "Waiting Time Mean": mean(waiting_times),
        "Waiting Time Median": median(waiting_times),
        "Waiting Time 99th Percentile": np.percentile(waiting_times, 99),
        "Processing Time Mean": mean(processing_times),
        "Processing Time Median": median(processing_times),
        "Processing Time 99th Percentile": np.percentile(processing_times, 99),
        "Tokens Per Second Mean": mean(tokens_per_second_list),
        "Tokens Per Second Median": median(tokens_per_second_list),
        "Tokens Per Second 99th Percentile": np.percentile(tokens_per_second_list, 99),
    }

    df = pd.DataFrame(resultsxlsx)
    output_file_xlsx = './chat_results_llama2_multithreaded.xlsx'
    with pd.ExcelWriter(output_file_xlsx, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

        pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name='Metrics')

    print(f"Results saved to {output_file_xlsx}")

    output_file_txt = './chat_results_llama2_multithreaded.txt'
    with open(output_file_txt, 'w') as f:
        f.write("Llama Model Results with Multithreading\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"PROMPT {result['Prompt Number']}:\n{result['Prompt']}\n")
            f.write(f"Generated Text:\n{result['Generated Text']}\n\n")
            f.write(f"Total Time: {result['Total Time (s)']:.2f} seconds\n")
            f.write(f"Waiting Time: {result['Waiting Time (s)']:.2f} seconds\n")
            f.write(f"Processing Time: {result['Processing Time (s)']:.2f} seconds\n")
            f.write(f"Tokens per Second: {result['Tokens per Second']:.2f}\n")
            f.write(f"Generated Tokens: {result['Generated Tokens']}\n")
            f.write(f"Finish Reason: {result['Finish Reason']}\n")
            f.write("\n" + "-" * 50 + "\n\n")

        f.write("\nSummary Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

    print(f"Results saved to {output_file_txt}")