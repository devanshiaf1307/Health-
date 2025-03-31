import google.generativeai as genai
from langchain.prompts import PromptTemplate
import gradio as gr
import pdfplumber

genai.configure(api_key="AIzaSyAbRxduuuxgTu-M9pbKBdKmPhgobl1YTZg")

model = genai.GenerativeModel("gemini-1.5-pro-latest")  

def extract_text(file):
    text = ""
    with pdfplumber.open(file.name) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        return "Error: No text found in the PDF. Ensure it's not a scanned image."

    prompt_template = PromptTemplate.from_template("""
    As a virtual doctor, analyze the medical report and provide a health diagnosis.
    
    **Medical Report:**
    ```
    {PDF_text}
    ```

    **Solution:**
    ```
    - Diagnosis & Explanation
    - Steps to Improve Health
    - Physical Activities & Dietary Recommendations
    ```
    """)

    prompt = prompt_template.format(PDF_text=text)

  
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Error: No response from AI."
    except Exception as e:
        return f"Error generating AI response: {str(e)}"


demo = gr.Interface(
    fn=extract_text,
    inputs="file",
    outputs="text",
    title="AI-Powered Health Report Analyzer",
    description="Upload your medical report, and AI will analyze it."
)

if __name__ == "__main__":
    demo.launch(debug=True)
