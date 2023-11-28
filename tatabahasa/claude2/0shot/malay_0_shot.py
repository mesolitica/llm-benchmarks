import sys
import os
from langchain.llms.bedrock import Bedrock
from tqdm import tqdm
import json
import random
import pickle as pkl
import concurrent.futures

def get_inference_parameters(): #return a default set of parameters based on the model's provider
        return { #anthropic
            "max_tokens_to_sample": 50,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"] 
           }
    
    
def get_text_response( input_content): #text-to-text client function
    
    model_kwargs = get_inference_parameters() #get the default parameters based on the selected model
    
    llm = Bedrock( #create a Bedrock llm client
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("us-east-1"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="anthropic.claude-v1", #use the requested model
        model_kwargs = model_kwargs
    )
    
    return llm.predict(f"\n\nHuman:I will ask you a question in Malay. Give me the correct option only and nothing else. \n\n{input_content}\n\nAssistant:") #return a response to the prompt

questions = []
with open('quiz-tatabahasa.jsonl') as fopen:
    for no, l in enumerate(fopen):
        l = json.loads(l)
        soalan = [l['question']]
        jawapan = None
        for c, k in l['choices'].items():
            soalan.append(f"{c}. {k['text']}")
            if k['answer']:
                jawapan = c
        
        data = {
            'no': no,
            'objektif': 'Jawab soalan yang diberikan' if l['instruction'] is None else l['instruction'],
            'soalan': '\n'.join(soalan),
            'jawapan': jawapan,
        }
        questions.append(data)
len(questions)

arange = set(range(len(questions)))

def convert_prompt(row, answer = False):
    if answer:
        prompt = f"""
objektif: {row['objektif']}
soalan: {row['soalan']}
jawapan: {row['jawapan']}
    """
    else:
        prompt = f"""
objektif: {row['objektif']}
soalan: {row['soalan']}
    """
    return prompt.strip()

i = 0
shots = random.sample([index for index in range(len(questions)) if index != i], 0)
prompts = []
for no, s in enumerate(shots):
    prompts.append(f'Contoh soalan {no + 1}\n' + convert_prompt(questions[s], answer = True))

prompts.append(convert_prompt(questions[i]))
prompt = '\n\n'.join(prompts)
print(prompt)

def process_question(i, questions):
    try:
        shots = random.sample([index for index in range(len(questions)) if index != i], 0)
        prompts = []

        for no, s in enumerate(shots):
            prompts.append(f'Contoh soalan {no + 1}\n' + convert_prompt(questions[s], answer=True))

        prompts.append(convert_prompt(questions[i]))
        prompt = '\n\n'.join(prompts)

        r = get_text_response(prompt)
        answer = r
        questions[i]['output'] = answer
    except Exception as e:
        # Handle exceptions here if needed
        print(f"Error processing question {i}: {e}")

num_workers = 2  # Adjust the number of threads as needed
# Assuming questions is a list of questions you want to process
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    list(tqdm(executor.map(lambda i: process_question(i, questions), range(len(questions))), total=len(questions)))

with open('save.pkl', 'wb') as f:
    pkl.dump(questions, f)

with open('save.pkl', 'rb') as f:
    questions = pkl.load(f)

import json
with open('output-0shot.json', 'w') as fopen:
    json.dump(questions, fopen)
filtered = [q for q in questions if 'output' in q]
correct = 0
for q in filtered:
    llm_output = q['output'].strip().split('.')[0]
    truth = q['jawapan']
    print(llm_output, truth, llm_output == truth)
    
    correct += llm_output == truth 
final_amount = (correct / len(filtered)) * 100
print(final_amount)



