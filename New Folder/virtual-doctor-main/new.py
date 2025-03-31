import google.generativeai as genai

genai.configure(api_key="AIzaSyAbRxduuuxgTu-M9pbKBdKmPhgobl1YTZg")  # Replace with your API key

try:
    available_models = genai.list_models()
    print([model.name for model in available_models])
except Exception as e:
    print(f"Error fetching models: {e}")
