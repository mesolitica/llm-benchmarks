import re
import os
import json
import random
from tqdm import tqdm
import anthropic
from anthropic import Anthropic


client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

def prompt_answer(input_, model_name="claude-3-opus-20240229",
                  temperature=0.2, top_p=1,
                  frequency_penalty=0, presence_penalty=0
                ):
    message = client.messages.create(
        max_tokens=4096,
        temperature=temperature,
        system="You are a helpful and smart malaysia assistant who answer only in Malay language.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_
                    }
                ]
            }
        ],
        model=model_name,
    )
    data = {
        'message': message.content[0].text,
        'finish_reason': message.stop_reason,
        'prompt': input_,
    }
    data['temperature'] = temperature
    data['top_p'] = top_p
    data['frequency_penalty'] = frequency_penalty
    data['presence_penalty'] = presence_penalty
    data['model_name'] = model_name
    return data

with open('BM-A-pt3') as fopen:
    text = fopen.read()
    
questions = []
for t in text.split('no: ')[1:]:
    t = t.strip()
    no = t.split('\n')[0]
    objektif = t.split('objektif: ')[1].split('\n')[0]
    soalan = t.split('soalan:')[1].split('jawapan:')[0].strip()
    jawapan = t.split('jawapan: ')[1].split(',')[0].strip()
    data = {
        'no': no,
        'objektif': objektif,
        'soalan': soalan,
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
Jawapan: {row['jawapan']}.
    """
    else:
        prompt = f"""
objektif: {row['objektif']}
soalan: {row['soalan']}
    """
    return prompt.strip()

for num_shot in [0,1,3]:

    model_name = "claude-3-opus-20240229"
    output_filename = 'bm_pt3-shot-{}-'.format(num_shot)+model_name+'.jsonl'
    inferenced = set()
    if os.path.exists(output_filename):
        with open(output_filename) as fopen:
            for no, l in enumerate(fopen):
                l = json.loads(l)
                inferenced.add(l['question']['no'])

    for idx, question in enumerate(tqdm(questions, dynamic_ncols=True)):
        if question['no'] in inferenced:
            continue

        prompt = convert_prompt(question)
        if num_shot > 0:
            shots = random.sample([index for index in range(len(questions)) if index != idx], num_shot)
            few_shots = []
            for order, shot in enumerate(shots):
                few_shots.append(f'Contoh soalan {order + 1}\n' + \
                                convert_prompt(questions[shot], answer=True)
                                )
            prompt = '\n\n'.join(few_shots)+'\n\nSekarang jawab soalan ini\n'+prompt
        response = prompt_answer(prompt, model_name=model_name, temperature=0)
        response['question'] = question
        with open(output_filename, 'a') as fout:
            fout.write(json.dumps(response) + '\n')

    correct, total = 0, 0
    ans_match = re.compile(r'Jawapan: [A-D].')
    base_answer = re.compile(r'[A-D]. ')
    ialah_answer = re.compile(r'ialah:\n[A-D].')
    with open(output_filename) as f:
        for line in f:
            payload = json.loads(line)
            msg = payload['message']
            if ans_match.search(msg):
                found = ans_match.search(msg).group(0)
                response = found.split('.')[0][-1]
            elif ialah_answer.search(msg):
                found = ialah_answer.search(msg).group(0)
                response = found.split('.')[0][-1]
            elif base_answer.search(msg):
                found = base_answer.search(msg).group(0)
                response = found.split('.')[0][-1]
            else:
                response = msg.split('.')[0][-1]
            if response not in ['A', 'B', 'C', 'D']:
                print(msg)
                print('----------------')
            label = payload['question']['jawapan']
            correct += response == label
            total += 1
    print(correct, total, 100*correct / total)