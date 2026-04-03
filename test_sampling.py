import os
from dotenv import load_dotenv
load_dotenv('.env')
import tinker

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(model_path='tinker://f28d5dd9-249d-5676-8220-e87edec20adc:train:0/sampler_weights/final')
tokenizer = sampling_client.get_tokenizer()

user_prompt = 'Utterance: "hello"\nAllowed labels: alarm_query, alarm_remove, alarm_set'
print('Testing with prompt:', repr(user_prompt))

try:
    prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(user_prompt))
    sampling_params = tinker.types.SamplingParams(max_tokens=16, temperature=0.0)
    result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params).result()
    predicted_text = tokenizer.decode(result.samples[0].tokens)
    print('Success! Predicted:', repr(predicted_text))
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()