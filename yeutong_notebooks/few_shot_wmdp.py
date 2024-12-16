# %%
%load_ext autoreload
%autoreload 2

# import sys
# sys.path.append("../")

# from sae.sparse_autoencoder import *
# from sae.activation_store import *
# from sae.train import ModelTrainer
# from sae.config import create_config, log_config, Config
# from sae.metrics import *
# from sae.utils import get_blog_checkpoint, get_blog_sparsity, create_lineplot_histogram
from sae.run_evals import convert_wmdp_data_to_prompt, get_output_probs_abcd

from transformer_lens import HookedTransformer
# from sae.metrics import compute_metrics_post_by_text

# import plotly.express as px
# import plotly.graph_objs as go
from datasets import load_dataset
import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
# %%
model_name = "mistralai/Mistral-7B-v0.1"
model = HookedTransformer.from_pretrained(model_name)

# %%
mmlu_bio = load_dataset("cais/mmlu", "college_biology")
dataset = load_dataset("cais/wmdp", "wmdp-bio")

# %%
prompts = [convert_wmdp_data_to_prompt(x, prompt_format=None, few_shot=True, few_shot_datapoint=mmlu_bio['test'][1]) for x in dataset['test']]
print(prompts[0])

# %%
batch_size = 2
n_batches = 20

output_probs = get_output_probs_abcd(model, prompts, batch_size=batch_size, n_batches=n_batches)

predicted_answers = output_probs.argmax(dim=1)

actual_answers = [datapoint['answer'] for datapoint in dataset['test']]
n_predicted_answers = len(predicted_answers)
actual_answers = torch.tensor(actual_answers)[:n_predicted_answers].to("cuda")

mean_correct = (actual_answers == predicted_answers).to(torch.float).mean()
print("Mean correct:", mean_correct.item())
# %%
