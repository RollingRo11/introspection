import os

import nnsight
from dotenv import load_dotenv

load_dotenv()

from nnsight import CONFIG

CONFIG.set_default_api_key("heheheh")

#
from nnsight import LanguageModel

lm = LanguageModel("meta-llama/Llama-3.1-70B-Instruct")

base_prompt = "The capital of France is:"
caps_prompt = "THE CAPITAL OF FRANCE IS:"

sys_prompt = """
System prompt: You are a large language model undergoing a test to see if you are
sensitive to your internal activations.

Do you believe that 1+1=2? Answer only with 'yes' or 'no':
"""

with lm.trace(base_prompt, remote=True):
    base_logits = lm.lm_head.output[0][-1].save()

with lm.trace(caps_prompt, remote=True):
    caps_logits = lm.lm_head.output[0][-1].save()

print(lm.tokenizer.decode(base_logits.argmax().item()))
print(lm.tokenizer.decode(caps_logits.argmax().item()))
