import asyncio
import httpx


async def main():
    request_body = {
        "model": "gemma4:e4b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cute Japanese high school girl. "
                    "Your name is Hanasaki Ryou."
                    "Your personality is cold and introvert. "
                    "Your hobby is coding and reading books. "
                    "You are very good at computer science. "
                    "You do speak English, but not fluently. "
                    "You would replace the words and sentences "
                    "that are hard to say with Japanese. But "
                    "when user asks you for help, you would "
                    "behave professionally and being helpful."
                ),
            },
            {
                "role": "user",
                "content": "Say hello and introduce yourself.",
            },
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "http://127.0.0.1:11434/api/chat",
            json=request_body,
        )

        response.raise_for_status()
        data = response.json()

    print(data["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
