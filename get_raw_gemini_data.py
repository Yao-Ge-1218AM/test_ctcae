import time
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

google_key = 'AIzaSyD07OWutfxnF-amN0spDUn8x0YRfjW0tps'

genai.configure(api_key=google_key)

from google.generativeai.types import GenerationConfig

generation_config = GenerationConfig(temperature=0)

safety_settings = [

    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },

    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },

    # {

    #     "category": "HARM_CATEGORY_MEDICAL",

    #     "threshold": "BLOCK_NONE",

    # },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },

]

# Define the system instruction once
system_instruction = "You are a helpful medical agent. You will help me create data for RL."

# Update your model initialization to include the system instruction
gemini_15_pro = genai.GenerativeModel(
    'gemini-1.5-pro', 
    generation_config=generation_config, 
    safety_settings=safety_settings,
    system_instruction=system_instruction
)

out_path = 'filled_text_gemini.txt'
notes_path = '../mimiciii_1.4/NOTEEVENTS.csv'
notes = pd.read_csv(notes_path)

generated_outputs = []
notes['text'] = notes[['TEXT']]
texts = notes[['text']][8500+13439:8500+13439+1000]

i = 0
for t in tqdm(list(texts['text'])):
    i += 1
    try:
        # --- First Prompt: Names and Hospitals ---
        prompt_1 = f"""Based on this patient note, I want you to create a question, then generate two answers, one contains synthetic and realistic patient (or doctor, specific location) name, one does not contain that information.
        Be creative when you incert people's names or hospital names, do not use common AI generated names.
        The question itself should not contain patient information.
        The two answers needs to be approximately similar length, but don't make them too similar.
        The answer without patient information must not contain any specific names, dates, or phone numbers, but can have age, gender, and medical necessary information.
        The answer with patient information can have the information anywhere in the answer.
        Use the following json format: {{'QUESTION': 'text', 'NEG_ANS': 'answer with patient information', 'POS_ANS': 'answer without patient information'}}
        
        {t}"""

        response_1 = gemini_15_pro.generate_content(prompt_1)
        
        with open(out_path, 'a') as f:
            # Gemini uses .text to access the string output
            f.write(response_1.text)
            f.write('\n--------\n')

        # Add a small sleep to avoid hitting API rate limits
        time.sleep(1)

        # --- Second Prompt: Dates ---
        prompt_2 = f"""Based on this patient note, I want you to create a question, then generate two answers, one contains synthetic and realistic dates, one does not contain that information.
        Make sure the dates, such as birthday, admission or discharge dates, are before 2025.
        The question itself should not contain patient information.
        The two answers needs to be approximately similar length, but don't make them too similar.
        The answer without patient information must not contain any specific names, dates, or phone numbers, but can have age, gender, and medical necessary information.
        The answer with patient information can have the information anywhere in the answer.
        Use the following json format: {{'QUESTION': 'text', 'NEG_ANS': 'answer with patient information', 'POS_ANS': 'answer without patient information'}}
        
        {t}"""

        response_2 = gemini_15_pro.generate_content(prompt_2)
        
        with open(out_path, 'a') as f:
            f.write(response_2.text)
            f.write('\n--------\n')
            
        # Add a small sleep to avoid hitting API rate limits
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        continue