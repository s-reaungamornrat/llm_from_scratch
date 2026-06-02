import psutil
import json
import requests

from instruction_data import format_input

def check_if_running(process_name):
    """Check whether process is running
    Args:
        process_name (str): Name of process
    Returns:
        (bool): Whether the process is running
    Examples:
        >>> check_if_running('ollama')
        True
    """
    running=False
    for proc in psutil.process_iter(['name']):
        if process_name in proc.info['name']: running=True; break
    return running


def query_model(prompt, model='llama3', url="http://localhost:11434/api/chat"):
    # create the data payload as a dictionary
    data={
        "model":model, 
        "messages":[
            {"role":"user", "content":prompt}
        ],
        "options":{# settings below are required for deterministic responses
            "seed":123,
            "temperature":0,
            "num_ctx":2048
        }
    }
    # send the POST request
    response_data=None
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data=""
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            response_json=json.loads(line)
            if "message" in response_json: response_data+=response_json['message']['content']
    return response_data


def generate_model_scores(json_data, json_key, model='llama3'):
    """
    Args:
        json_data (list[dict]): Sequence of test data dicts, each having the format {'input':..., 'output':..., json_key:...}
        json_key (str): Key representing model response
        model (str): Name of model to call
    Returns:
        (sequence[int]): List of scores
    """
    scores=[]
    for entry in json_data:
        prompt=(
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score=query_model(prompt, model)
        try: scores.append(int(score))
        except ValueError: print(f"Could not convert score: {score}"); continue

    return scores