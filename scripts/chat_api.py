import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Chat with a local DroneMind OpenAI-compatible server")
    parser.add_argument("--base_url", default="http://localhost:8998/v1")
    parser.add_argument("--api_key", default="sk-123")
    parser.add_argument("--model", default="dronemind")
    parser.add_argument("--stream", type=int, default=1, choices=[0, 1])
    parser.add_argument("--history", type=int, default=0, help="Number of recent messages to send; 0 sends only the latest user message")
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    conversation_history = []
    stream = bool(args.stream)

    print(f"DroneMind API: {args.base_url}")
    print("Press Ctrl+C to exit.\n")

    while True:
        query = input("[Q]: ").strip()
        if not query:
            continue

        conversation_history.append({"role": "user", "content": query})
        messages = conversation_history[-(args.history or 1):]
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            stream=stream,
            temperature=0.8,
            max_tokens=2048,
            top_p=0.8,
            extra_body={"chat_template_kwargs": {"open_thinking": bool(args.open_thinking)}},
        )

        if not stream:
            assistant_res = response.choices[0].message.content or ""
            print("[A]: ", assistant_res)
        else:
            print("[A]: ", end="", flush=True)
            assistant_res = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = delta.content or ""
                if reasoning:
                    print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
                if content:
                    print(content, end="", flush=True)
                assistant_res += content

        conversation_history.append({"role": "assistant", "content": assistant_res})
        print("\n")


if __name__ == "__main__":
    main()
