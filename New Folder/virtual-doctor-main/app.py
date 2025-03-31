import google.generativeai as genai
from langchain.prompts import PromptTemplate
import gradio as gr
import pdfplumber

api_key = 'AIzaSyAbRxduuuxgTu-M9pbKBdKmPhgobl1YTZg' # https://makersuite.google.com/
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-pro')

def extract_text(file_path):
    with open(file_path, 'rb') as f:
        pdf = pdfplumber.open(f)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        prompt_template = PromptTemplate.from_template("""
        As a virtual doctor, I'm here to assist you with your health journey. 
        Please provide your medical report, and 
        I will analyze my medical reports and guide me intelligently so that my help can also improve.
        Based on your report I tell you  what is my problem including your name at the top of the solution
        provides solution to recover from the disease?
        What physical activities should I do, what habits should I give up and what fruits should I eat? 
        So that my health can improve and I can remain healthy in future and my disease also goes away.
        

        **medical report:**
        ```
        {PDF_text}
        ```
        **Solution:**
        ```
        [Notes and important information]
        ```"""
        )
        prompt = prompt_template.format(PDF_text=text)
        return model.generate_content(prompt).text
demo = gr.Interface(
    fn=extract_text,
    inputs="file",
    outputs="text",
    description="Personalized Guidance for Your Well-being Journey",
    )

demo.launch(debug=True)