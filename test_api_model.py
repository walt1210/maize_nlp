from google import genai

import config

# Migrated from google.generativeai (deprecated, all support ended) to
# google.genai — same pattern as the rest of the pipeline now.
client = genai.Client(api_key=config.GEMINI_API_KEY)

for model_name in [config.GEMINI_MODEL, config.GEMINI_MODEL_LITE]:
    try:
        response = client.models.generate_content(model=model_name, contents="Say OK")
        print(f"{model_name}: WORKS — {response.text.strip()}")
    except Exception as exc:
        print(f"{model_name}: FAILED — {exc}")
