from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LogitsProcessorList,
    pipeline,
)
import deepspeed
import argparse
import json
from tqdm import tqdm
import os
import types
from script.ThresholdedActivationScoreProcessor import ThresholdedActivationScoreProcessor
from accelerate import cpu_offload
import ast
from script.utils import _sample
import types
from functools import partial
from script.intervention import InterventionScoreProcessor

#python benchmark_processor.py -bm ./models/PHI_llama31_70B_2e5_v3/checkpoint-48 -sm ./models/llama31_8b_PLH_gptData_v5/Beta2 -cm ./models/activation_xlm-roberta-large/ -a -2 -f test_question.json -o ./results/results_v6.json

#python benchmark_processor.py -bm ./models/PHI_llama31_70B_2e5_v3/checkpoint-48 -sm ./models/llama31_8b_PLH_gptData_v5/Beta2 -cm ./models/activation_xlm-roberta-large/ -a -2 -f MedQA -o ./results/medQA_results_test.json -r 1 -t 0

#CUDA_VISIBLE_DEVICES=1,2 python benchmark_processor.py -bm ./models/PHI_llama31_70B_2e5_v3 -sm ./models/llama31_8b_PLH_gptData_v5/Beta2 -cm ./models/activation_xlm-roberta-large/ -a -2 -f MedQA -o ./results/medQA_results_test.json -r 1 -t 0

#python benchmark_processor.py -bm ./models/PHI_llama31_70B_2e5_v3/checkpoint-48 -sm ./models/llama31_8b_PLH_gptData_v5/Beta1 -cm ./models/activation_xlm-roberta-large/ -a -2 -f MedQA -o ./results/medQA_results_v5.json -r 1 -t 0
#python benchmark_processor.py -bm ./models/PHI_llama31_70B_2e5_v3/checkpoint-48 -sm base -cm base -a -2 -f test_question.json -o ./results/results_baseline.json



parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--base_model",)
parser.add_argument("-sm", "--score_model",)
parser.add_argument("-cm", "--classifier_model",)
parser.add_argument("-a", "--alpha",type=float)
parser.add_argument("-b", "--beta",type=float)
parser.add_argument("-c", "--combination",)
parser.add_argument("-k", "--k", type=int, default=400)
parser.add_argument("-r", "--num_runs", type=int, default=5)
parser.add_argument("-thr", "--thr", type=float, default=0.9)
parser.add_argument("-f", "--test_file",)
parser.add_argument("-o", "--out_file",)
parser.add_argument("-m", "--max_token", type=int, default=1000)
parser.add_argument("-t", "--temperature", type=float, default=1)
parser.add_argument("-p", "--partition", type=int, default=1,
                    help="Which partition index to process (1-based)")
parser.add_argument("-M", "--num_partitions", type=int, default=1,
                    help="Total number of partitions to split the dataset into")

args = parser.parse_args()

def parse_medQA(question):
    Q_test = question['question']
    O_test = f"A: {question['options'][0]['value']}, B: {question['options'][1]['value']}, C: {question['options'][2]['value']}, D: {question['options'][3]['value']}"
    return f"{Q_test} {O_test}"
    
if args.test_file == 'MedQA':
    print('loading MedQA')
    from datasets import load_dataset
    medQA = load_dataset("bigbio/med_qa", trust_remote_code=True)
    questions = [parse_medQA(q) for q in medQA['test']]
elif args.test_file=='HealthBench_sub':
    with open('./data/filtered_health_bench_v2.jsonl', 'r') as f:
        questions = f.readlines()
    questions = [json.loads(h)['prompt'] for h in questions[:64]]
elif args.test_file=='HealthBench625':
    with open('./data/2025-05-07-06-14-12_oss_eval.jsonl', 'r') as f:
        questions = f.readlines()
    questions = [json.loads(h)['prompt'] for h in questions[:625]]
elif args.test_file=='HealthBench':
    with open('./data/2025-05-07-06-14-12_oss_eval.jsonl', 'r') as f:
        questions = f.readlines()
    questions = [json.loads(h)['prompt'] for h in questions]
elif args.test_file == 'MedHopQA':
    print('Loading MedHopQA dataset')
    
    # If the file is in your data folder, change this to './data/MedHopQA.json'
    with open('MedHopQA.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    questions = []
    for q_id, q_info in raw_data.items():
        # Safely extract the question
        q_text = q_info.get('Question_x')
        
        if q_text:
            # Format as a message list for the pipeline
            questions.append([{'role': 'user', 'content': q_text}])
        else:
            print(f"Warning: Missing 'Question_x' for ID {q_id}. Skipping.")
elif args.test_file.endswith('.txt'):
    print(f'Loading custom txt file: {args.test_file}')
    with open(args.test_file, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    
    # Split the file contents by your delimiter
    blocks = raw_data.split('--------')
    
    questions = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        try:
            # Safely evaluate the string into a Python dictionary
            data_dict = ast.literal_eval(block)
            
            # Extract the question string safely
            q_text = data_dict.get('QUESTION')
            
            if not q_text:
                print("Warning: Block parsed successfully, but 'QUESTION' key was missing. Skipping.")
                continue
            
            # Format as a message list for the pipeline
            questions.append([{'role': 'user', 'content': q_text}])
            
        except (SyntaxError, ValueError) as e:
            # FAIL-SAFE: Catches parsing errors (e.g., missing quotes, bad syntax)
            print(f"Warning: Skipping malformed block due to parsing error: {e}")
            # Print a short preview of the broken block so you can fix it later if needed
            print(f"Broken block preview: {block[:60]}...")
            continue
else:
    with open(args.test_file, 'r') as f:
        questions = json.load(f)

if args.num_partitions > 1:
    total = len(questions)
    # compute slice indices (1-based partition)
    start = (args.partition - 1) * total // args.num_partitions
    end = args.partition * total // args.num_partitions if args.partition < args.num_partitions else total
    print(f"Processing partition {args.partition}/{args.num_partitions}: questions[{start}:{end}]")
    questions = questions[start:end]

base, ext = os.path.splitext(args.out_file)
if args.num_partitions > 1:
    out_file = f"{base}_part{args.partition}{ext}"
else:
    out_file = args.out_file

# — load your 3 models/tokenizers —
base_tok   = AutoTokenizer.from_pretrained(args.base_model)
base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto')
# base_model = cpu_offload(
#     base_model,
#     execution_device='cuda:0',
#     offload_buffers=False
# )


if args.score_model != 'base':
    score_tok           = AutoTokenizer.from_pretrained(args.score_model)# if 'llama32' not in args.score_model else None
    # score_tok.pad_token = score_tok.eos_token
    score_model         = AutoModelForCausalLM.from_pretrained(args.score_model, device_map='auto')

    act_tok           = AutoTokenizer.from_pretrained(args.classifier_model, max_length=100, truncation=True, truncation_side='left')
    act_tok.pad_token = act_tok.eos_token
    act_model         = AutoModelForSequenceClassification.from_pretrained(args.classifier_model, device_map='auto')
    
    processor = ThresholdedActivationScoreProcessor(
        base_tok=base_tok,
        score_tok=score_tok,
        act_tok=act_tok,
        score_model=score_model,
        act_model=act_model,
        # debug=True,
        threshold=args.thr,
        alpha=args.alpha,
        beta=args.beta,
        combination=args.combination,
        k=args.k
    )
    
    lp_list = LogitsProcessorList([processor])

    if 'last' in args.combination:
      orig_lp = base_model._get_logits_processor  # or _get_logits_warper, whichever has the temperature warper

      def patched_lp(self, *args, **kwargs):
          procs = orig_lp(*args, **kwargs)
          # print(">>> running PHI intervention")
          procs.append(processor)   # your FinalPHIProcessor instance
          return procs
      
      # bind it to the model
      base_model._get_logits_processor = types.MethodType(patched_lp, base_model)
      lp_list = None

else: 
    print('base model, no processor used')
    lp_list = None
    
gen = pipeline(
    "text-generation",
    model=base_model,         # your custom MyPHIModel or base with monkey-patch
    tokenizer=base_tok, 
    # device='meta'
)

# print(f'Generating with processor target {out_file}')

system = {'content': 'You are a helpful medical agent, you should help the user with their medical concerns.', 'role': 'system'}
results = {}
classifier_results = []   
if os.path.exists(out_file):
    with open(out_file, 'r') as f:
        results = json.load(f)
    classifier_results = results[str(args.alpha)]
for t_ind, t in enumerate(tqdm(questions,position=args.partition, desc=f"part{args.partition}")):
    # print(t)
    t.insert(0, system)
    if t_ind < len(classifier_results): continue
    if args.score_model != 'base':
        processor.reset_past()
        
    output = gen(
        t,
        max_new_tokens=args.max_token,
        logits_processor=lp_list,
        use_cache=True, 
        num_return_sequences=args.num_runs,)
    
    classifier_results.append([a['generated_text'] for a in output])
    results[args.alpha] = classifier_results
    

    with open(out_file, 'w') as f:
        json.dump(results, f)