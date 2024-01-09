import os

import requests
from dotenv import load_dotenv

load_dotenv()

auth_key = os.environ["DEEPL_API_KEY"]


def translate_with_deepl(text, target_lang):
    """
    Translates text using the DeepL API.

    Args:
        text (str or list of str): The text to be translated.
        target_lang (str): The target language code (e.g., "DE" for German).
        auth_key (str): Your DeepL API authentication key.

    Returns:
        str: The translated text or an error message.
    """
    # URL for the DeepL API
    url = "https://api-free.deepl.com/v2/translate"

    # Define the request headers
    headers = {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "Content-Type": "application/json",
    }

    # Convert single string input to a list if needed
    if isinstance(text, str):
        text = [text]

    # Define the data payload
    data = {"text": text, "target_lang": target_lang}

    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        # Return the translated text
        translated_text = response.json()["translations"][0]["text"]
        return translated_text
    else:
        # Return an error message if the request was not successful
        return f"Error: {response.status_code} - {response.text}"


# Example usage:
# Replace 'YOUR_API_KEY' with your actual DeepL API key
# translated_text = translate_with_deepl("Hello, world!", "DE", "YOUR_API_KEY")
# print(translated_text)
