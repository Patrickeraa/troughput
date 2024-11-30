import requests
import time
import pandas as pd
from statistics import mean, median
import numpy as np

SERVER_URL = "http://server:8050/process-prompt/"

results = []
resultsxlsx = []

def send_prompt(prompt: str):
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

        result.update({
            "total_time": total_time,
            "waiting_time": waiting_time,
            "tokens_per_second": tokens_per_second,
        })
        return result
    else:
        return {"error": f"Server returned status code {response.status_code}", "details": response.text}

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

    print("Sending prompts to the Llama model...\n")

    for prompt_number, prompt in enumerate(prompts, start=1):
        print(f"Sending prompt {prompt_number}: {prompt}")
        result = send_prompt(prompt)

        if "error" not in result:
            response_text = result["response"]
            total_time = result["total_time"]
            processing_time = result["processing_time"]
            waiting_time = result["waiting_time"]
            tokens_per_second = result["tokens_per_second"]
            generated_tokens = result["generated_tokens"]
            finish_reason = result["finish_reason"]

            results.append({
                "Prompt Number": prompt_number,
                "Prompt": prompt,
                "Generated Text": response_text,
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

            print(f"Llama: {response_text}")
            print(f"[Total Time: {total_time:.2f}s | Waiting Time: {waiting_time:.2f}s | "
                  f"Processing Time: {processing_time:.2f}s | Tokens per Second: {tokens_per_second:.2f}]")
        else:
            print(f"Error: {result['error']}")
            print(f"Details: {result['details']}")

        print("-" * 50)

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

    # Save results to .xlsx
    df = pd.DataFrame(resultsxlsx)
    output_file_xlsx = './chat_results_llama2.xlsx'
    with pd.ExcelWriter(output_file_xlsx, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        # Save Metrics Sheet
        pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name='Metrics')

    print(f"Results saved to {output_file_xlsx}")

    # Save results to .txt
    output_file_txt = './chat_results_llama2.txt'
    with open(output_file_txt, 'w') as f:
        f.write("Llama Model Results\n")
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



