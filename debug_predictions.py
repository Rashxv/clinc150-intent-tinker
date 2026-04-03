import os
from dotenv import load_dotenv
load_dotenv('.env')
import tinker
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.dataset_utils import read_jsonl

print("Loading data...")
rows = read_jsonl(Path('data/processed/val.jsonl'))[:1]
print(f"Loaded {len(rows)} rows")

print("Creating service client...")
service_client = tinker.ServiceClient()
print("Creating sampling client...")
sampling_client = service_client.create_sampling_client(model_path='tinker://f28d5dd9-249d-5676-8220-e87edec20adc:train:0/sampler_weights/final')
print("Getting tokenizer...")
tokenizer = sampling_client.get_tokenizer()
print("Tokenizer obtained")

# Extract prompt
user_prompt = None
for msg in rows[0]['messages']:
    if msg['role'] == 'user':
        user_prompt = msg['content']
        break

print(f"User prompt: {user_prompt[:100]}...")

print("Encoding prompt...")
prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(user_prompt))
print("Creating sampling params...")
sampling_params = tinker.types.SamplingParams(max_tokens=16, temperature=0.0)
print("Sampling...")
result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params).result()
print("Decoding result...")
predicted_text = tokenizer.decode(result.samples[0].tokens)
print(f"Success! Predicted: {predicted_text}")