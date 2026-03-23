# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
from typing import Dict, List


from mathruler.grader import extract_boxed_content, grade_answer


import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import re
import json
import os


def format_reward(predict: str) -> float:
   pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
   format_match = re.fullmatch(pattern, predict)
   return 1.0 if format_match else 0.0




def accuracy_reward(predict: str, ground_truth: str) -> float:
   answer = extract_boxed_content(predict)
   return 1.0 if grade_answer(answer, ground_truth) else 0.0


def is_number(s):
   try:
       float(s)  # Will handle both int and float strings
       return True
   except ValueError:
       return False
  
def similarity_score(a, b):
   if a == b:
       return 1.0
   if a == 0 or b == 0:
       return 0.0
   return 1 - (abs(a - b) / max(abs(a), abs(b)))




async def fetch(session, prompt, semaphore, url, headers=None, openrouter_model=None):
   async with semaphore:
       if headers is not None and openrouter_model is not None:
           payload = {
               "model": openrouter_model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7,
               "max_tokens": 100,
               "stream": False,
           }
           async with session.post(url, json=payload, headers=headers) as response:
               if response.status == 200:
                   raw = await response.json()
                   content = ""
                   try:
                       choices = raw.get("choices") if isinstance(raw, dict) else None
                       if isinstance(choices, list) and choices:
                           msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                           content = str(msg.get("content") or "")
                   except Exception:
                       content = ""
                   return {"text": [content]}
               return {"error": response.status, "text": [await response.text()]}

       payload = {
           "prompt": prompt,
           "stream": False,
           "temperature": 0.7,
           "max_tokens": 100
       }
       async with session.post(url, json=payload) as response:
           if response.status == 200:
               result = await response.json()
               return result
           return {"error": response.status, "text": [await response.text()]}


async def main(prompts, url, headers=None, openrouter_model=None):
   CONCURRENCY = 100
   semaphore = asyncio.Semaphore(CONCURRENCY)
   async with aiohttp.ClientSession() as session:
       tasks = [fetch(session, prompt, semaphore, url, headers=headers, openrouter_model=openrouter_model) for prompt in prompts]
       results = await asyncio.gather(*tasks)
       return results


def batch_process(batch):
   with open('./judge/judge_info.json', 'r') as file:
       data = json.load(file)
      
   host = data.get('host')
   openrouter_api_key = (data.get("openrouter_api_key") or data.get("openrouter_key") or "").strip()
   openrouter_model = (data.get("openrouter_model") or "deepseek/deepseek-v3.2").strip()
   openrouter_base_url = (data.get("openrouter_base_url") or "https://openrouter.ai/api/v1/chat/completions").strip()
   openrouter_http_referer = (data.get("openrouter_http_referer") or "").strip()
   openrouter_app_title = (data.get("openrouter_app_title") or "vtool_r1_judge").strip()


   headers = None
   if openrouter_api_key:
        url = openrouter_base_url
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openrouter_api_key}"}
        if openrouter_http_referer:
            headers["HTTP-Referer"] = openrouter_http_referer
        if openrouter_app_title:
            headers["X-Title"] = openrouter_app_title
   else:
        if os.environ.get("NGROK") == "YES":
             url = "your ngrok domain"
        else:
             url = f"http://{host}:7999/generate"

   #CONCURRENCY = 100  # Limit to avoid overwhelming the server
   #semaphore = asyncio.Semaphore(CONCURRENCY)
   #results = asyncio.run(main(batch, semaphore, url))
   results = asyncio.run(main(batch, url, headers=headers, openrouter_model=openrouter_model if headers is not None else None))
   return results


def extract_result(text):
   # Extract the last occurrence of content inside <>
   match = re.findall(r'<(.*?)>', text)


   if match:
       last_value = match[-1]
       return last_value == "|YES|"
  
   return False


def compute_score(predicts: List[str], ground_truths: List[str], queries: List[str], penalties: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
   with open("./judge/judge_prompt.txt", 'r') as file:
       judge_prompt = file.read()


   scores = []


   evaluate_indices = []
   evaluate_batch = []
  
   for idx, (predict, ground_truth, query, penalty) in enumerate(zip(predicts, ground_truths, queries, penalties)):


       overall_score = 0
  
       answers = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)', predict)
       if len(answers) > 0:
           answers = answers[0]
           evaluate_prompt = judge_prompt.replace("<question>", query).replace("<gt>", ground_truth).replace("<answer>", answers)
           evaluate_indices.append(idx)
           evaluate_batch.append(evaluate_prompt)


       else:
           overall_score = 0


           #we calculate here
           #this is for INCORRECT


           '''if penalty != 0: #tool has been used
               overall_score = -0.5'''


       scores.append(
           {
               "overall": 0,
               #"penalty": penalty,
               "accuracy": 0
           }
       )


   results = batch_process(evaluate_batch)


   for idx, result in enumerate(results):
       penalty = penalties[evaluate_indices[idx]]
       response = result["text"][0]
       if extract_result(response):
           #if no "penalty" is applied, not like we are adding penalty, currently treat it like a flag for "invalid tool use"
           '''if penalties[evaluate_indices[idx]] == 0:
               scores[evaluate_indices[idx]]["overall"] += 1'''
           '''if penalty == 1: #tool use correct
               scores[evaluate_indices[idx]]["overall"] = 1.5
           elif penalty == -1: #tool use incorrect
               scores[evaluate_indices[idx]]["overall"] = 0
           else: # no tools used
               scores[evaluate_indices[idx]]["overall"] = 1'''
              
           scores[evaluate_indices[idx]]["overall"] = 1
           scores[evaluate_indices[idx]]["accuracy"] = 1
       else:
           '''if penalty != 0: #tool has been used
               scores[evaluate_indices[idx]]["overall"] = -0.5'''


       #response = result["text"][0]
       #print(extract_assistant_response(response))


   return scores


def compute_score_jc(predicts: List[str], ground_truths: List[str], queries: List[str], penalties: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
   with open("./judge/judge_prompt.txt", 'r') as file:
       judge_prompt = file.read()


   scores = []


   evaluate_indices = []
   evaluate_batch = []
  
   for idx, (predict, ground_truth, query, penalty) in enumerate(zip(predicts, ground_truths, queries, penalties)):
       answers = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)', predict)
       if len(answers) > 0:
           answers = answers[0]
           evaluate_prompt = judge_prompt.replace("<question>", query).replace("<gt>", ground_truth).replace("<answer>", answers)
           evaluate_indices.append(idx)
           evaluate_batch.append(evaluate_prompt)


       else:
           overall_score = 0


       scores.append(
           {
               "overall": 0,
               #"penalty": penalty,
               "accuracy": 0
           }
       )


   results = batch_process(evaluate_batch)


   for idx, result in enumerate(results):
       penalty = penalties[evaluate_indices[idx]]
       response = result["text"][0]
       if extract_result(response):
           #if no "penalty" is applied, not like we are adding penalty, currently treat it like a flag for "invalid tool use"
           '''if penalties[evaluate_indices[idx]] == 0:
               scores[evaluate_indices[idx]]["overall"] += 1'''
           '''if penalty == 1: #tool use correct
               scores[evaluate_indices[idx]]["overall"] = 1.5
           elif penalty == -1: #tool use incorrect
               scores[evaluate_indices[idx]]["overall"] = 0
           else: # no tools used
               scores[evaluate_indices[idx]]["overall"] = 1'''
              
           scores[evaluate_indices[idx]]["overall"] = 1


           if penalty == 1: #tool use correct
               scores[evaluate_indices[idx]]["overall"] = 1.5


           scores[evaluate_indices[idx]]["accuracy"] = 1
       else:
           '''if penalty != 0: #tool has been used
               scores[evaluate_indices[idx]]["overall"] = -0.5'''


       #response = result["text"][0]
       #print(extract_assistant_response(response))


   return scores




def compute_score_double(predicts: List[str], ground_truths: List[str], queries: List[str], penalties: List[str], rollout_rounds: List[str], ids: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
   with open("./judge/judge_prompt.txt", 'r') as file:
       judge_prompt = file.read()


   scores = []


   evaluate_indices = []
   evaluate_batch = []


   with_second_rollouts = set()


   for idx in range(len(predicts)):
       if rollout_rounds[idx] == 1:
           #second rollout
           with_second_rollouts.add(ids[idx])


   reward_by_id = {}
  
   for idx, (predict, ground_truth, query, penalty, rollout_round, id) in enumerate(zip(predicts, ground_truths, queries, penalties, rollout_rounds, ids)):


       if id in with_second_rollouts and rollout_round == 0:
           #this one has second rollout but isn't the second one, we aren't evaluating this but using the answer/reward from its second rollout
           continue


       answers = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)', predict)
       if len(answers) > 0:
           answers = answers[0]
           evaluate_prompt = judge_prompt.replace("<question>", query).replace("<gt>", ground_truth).replace("<answer>", answers)
           evaluate_indices.append(idx)
           evaluate_batch.append(evaluate_prompt)


       reward_by_id[id] = { "overall": 0, "accuracy": 0, "ignore": 0 }


       '''scores.append(
           {
               "overall": 0,
               #"penalty": penalty,
               "accuracy": 0
           }
       )'''


   results = batch_process(evaluate_batch)


   for idx, result in enumerate(results):
       penalty = penalties[evaluate_indices[idx]]
       response = result["text"][0]
       if extract_result(response):
           id = ids[evaluate_indices[idx]]
           reward_by_id[id]["overall"] = 1
           reward_by_id[id]["accuracy"] = 1
       else:
           '''if penalty != 0: #tool has been used
               scores[evaluate_indices[idx]]["overall"] = -0.5'''


       #response = result["text"][0]
       #print(extract_assistant_response(response))


   scores = []


   for idx, id in enumerate(ids):
       #so we know what to ignore
       if id in with_second_rollouts and rollout_rounds[idx] == 0:
           k = reward_by_id[id].copy()
           #k["ignore"] = 1
           k["overall"] *= 5
           scores.append(k)
       #this may work since it does not screw anything significantly
       elif id in with_second_rollouts and rollout_rounds[idx] == 1:
           k = reward_by_id[id].copy()
           #k["overall"] *= 10
           scores.append(k)
       elif penalties[idx] == -10:
           #we penalize this situation
           k = reward_by_id[id].copy()
           k["overall"] = 0
           k["accuracy"] = 0
           scores.append(k)
       else:
           scores.append(reward_by_id[id])


   return scores
