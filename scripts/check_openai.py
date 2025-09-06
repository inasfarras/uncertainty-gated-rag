from agentic_rag.config import settings
from openai import AuthenticationError, OpenAI


def check_openai_api_key():
    """
    Checks if the OpenAI API key is set and valid.
    """
    api_key = settings.OPENAI_API_KEY

    if not api_key:
        print("❌ Error: OPENAI_API_KEY is not set in your environment.")
        print("Please set it in your .env file or as an environment variable.")
        return

    print("🔑 OPENAI_API_KEY found.")

    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        print("✅ OpenAI API key is valid and working.")
    except AuthenticationError:
        print("❌ Error: The provided OpenAI API key is invalid.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    check_openai_api_key()
