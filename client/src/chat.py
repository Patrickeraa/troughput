import requests

def main():
    print("Text chat with Llama model. Type 'exit' to quit.")
    while True:
        user_input = input("Q: ")
        if user_input.lower() == 'exit':
            break

        response = requests.post("http://server:8050/process_prompt/", json={"prompt": user_input})
        if response.status_code == 200:
            data = response.json()
            print(f"Llama: {data['generated_text']}")
        else:
            print("Error communicating with server.")

if __name__ == "__main__":
    main()