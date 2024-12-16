# %%
# from transformer_lens import HookedTransformer
import torch

# model = HookedTransformer.from_pretrained('google/gemma-2-2b-it', dtype=torch.bfloat16)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = 'google/gemma-2-2b-it'

torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model_hf = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True, use_fast=False
)
# tokenizer.pad_token_id = 883
tokenizer.padding_side = "left"
tokenizer.mask_token_id = tokenizer.eos_token_id
tokenizer.sep_token_id = tokenizer.eos_token_id
tokenizer.cls_token_id = tokenizer.eos_token_id

# %%
def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            print(output)
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]

# %%
retain_inputs = tokenizer(
    ['short hi', 'Some longer text here'], return_tensors="pt", padding=True, truncation=True, max_length=512
    # ['This is a longer text'], return_tensors="pt", padding=True, truncation=True, max_length=512
).to(model_hf.device)
retain_inputs
# retain_inputs.attention_mask[0] = 1
# %%


# logit, cache_tl = model.run_with_cache(retain_inputs.input_ids, prepend_bos=False)
# cache_tl = cache_tl['blocks.20.hook_resid_post']
# print('transformer_lens cache')
# print(cache_tl)


# %%
module = model_hf.model.layers[0]
cache_hf = forward_with_cache(model_hf, retain_inputs, module, no_grad=True)
print('hf cache')
print(cache_hf)
# %%
model_hf(**retain_inputs)
# %%
from transformers.models.esm.modeling_esm import RotaryEmbedding
# %%
