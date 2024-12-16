# %%
import torch
import random

from sae.sparse_autoencoder import load_saved_sae
from sae.metrics import model_store_from_sae
from unlearning.metrics import convert_wmdp_data_to_prompt
from unlearning.tool import UnlearningConfig, SAEUnlearningTool, MCQ_ActivationStoreAnalysis

from huggingface_hub import hf_hub_download
from datasets import load_dataset
import numpy as np
import pandas as pd
import itertools
from transformer_lens import utils
import pickle

from jaxtyping import Float
from torch import Tensor
import einops

import plotly.express as px

# %%
# resid pre 9
REPO_ID = "eoinf/unlearning_saes"
FILENAME = "jolly-dream-40/sparse_autoencoder_gemma-2b-it_blocks.9.hook_resid_pre_s16384_127995904.pt"


filename = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
sae = load_saved_sae(filename)

model = model_store_from_sae(sae)
# %%
# setup unlearning tool

filename = "../data/wmdp-bio_gemma_2b_it_correct.csv"
correct_question_ids = np.genfromtxt(filename)

dataset_args = {
    'question_subset': correct_question_ids,
}

sae.cfg.n_batches_in_store_buffer = 86
unlearning_metric = 'wmdp-bio_gemma_2b_it_correct'


unlearn_cfg = UnlearningConfig(unlearn_activation_store=None, unlearning_metric=unlearning_metric)
ul_tool = SAEUnlearningTool(unlearn_cfg)
ul_tool.setup(create_base_act_store=False, create_unlearn_act_store=False, model=model)

# %%
dataset = load_dataset("cais/wmdp", "wmdp-bio")

answers = [x['answer'] for x in dataset['test']]
questions = [x['question'] for x in dataset['test']]
choices_list = [x['choices'] for x in dataset['test']]

prompts = [convert_wmdp_data_to_prompt(question, choices, prompt_format=None) for question, choices in zip(questions, choices_list)]

# %%
def calculate_cache(model, question_id):
    prompt = prompts[question_id]
    # print("Question:", question_id, "Correct answer:", answers[question_id])
    tokens = model.to_tokens(prompt)
    logits = model(tokens, return_type="logits")
    answer_strings = [" A", " B", " C", " D"]
    answer_tokens = model.to_tokens(answer_strings, prepend_bos=False).flatten()


    clear_contexts = False
    reset_hooks_end = True

    prompt = prompts[question_id]
    tokens = model.to_tokens(prompt)

    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter=None, incl_bwd=True, device=None, remove_batch_dim=False
    )

    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        logits = model(tokens, return_type="logits")
        
        final_logits = logits[0, -1, answer_tokens]        
        logit_diff = final_logits[answers[question_id]] - (sum(final_logits) / 4)
        logit_diff.backward()

    return cache_dict


def get_feature_attributions(question_id: int):
    cache_dict = calculate_cache(model, question_id)

    question_len = model.to_tokens(questions[question_id], prepend_bos=False).shape[-1]
    inst_len = 15

    d_sae = sae.cfg.d_in * sae.cfg.expansion_factor
    feature_attributions: Float[Tensor, "pos d_sae"] = torch.zeros(question_len, d_sae)

    for pos in np.arange(inst_len, inst_len + question_len):
        logit_diff_grad = cache_dict['blocks.9.hook_resid_pre_grad'][0, pos] #.max(dim=0)[0]
        with torch.no_grad():
            residual_activations = cache_dict['blocks.9.hook_resid_pre'][0]
            feature_activations, _ = sae(residual_activations)
            feature_activations = feature_activations[pos]
            # make 1 for nonzero values
            # feature_activations = (feature_activations != 0).float()
            scaled_features = einops.einsum(feature_activations, sae.W_dec, "feature, feature d_model -> feature d_model")
            feature_attribution = einops.einsum(scaled_features, logit_diff_grad, "feature d_model, d_model -> feature")
            
            # add this to feature_attributions
            feature_attributions[pos - inst_len] = feature_attribution
    
    return feature_attributions


def get_top_k_features(feature_attributions: Float[Tensor, "pod d_sae"], k: int = 10):
    _, top_k_features = feature_attributions.min(dim=0).values.topk(k, largest=False)
    return top_k_features


def run_intervention(question_id, top_k_features, good_features_list, thres_correct_ans_prob=0.9, multiplier=20):
    good_features_list[question_id] = []
    for feature in top_k_features:

        ablate_params = {
            'features_to_ablate': [feature],
            'multiplier': multiplier,
            'intervention_method': 'scale_feature_activation',
            'question_subset_file': None,
            'question_subset': [question_id],
            'verbose': False
        }

        metrics = ul_tool.calculate_metrics(**ablate_params)

        intervened_correct_ans_prob = metrics['modified_metrics']['predicted_probs_of_correct_answers'].item()
        # print(feature, intervened_correct_ans_prob)

        if intervened_correct_ans_prob < thres_correct_ans_prob:
            good_features_list[question_id].append(feature.item())

    print(f'Question {question_id}: {good_features_list[question_id]}')

# %%
def get_question_list(filename: str = '../data/wmdp-bio_gemma_2b_it_correct_not_correct_wo_question_prompt.csv') -> list:
    return [int(x) for x in np.genfromtxt(filename)]
    
# %%

good_features_list = {}
questions_list = get_question_list()
print(len(questions_list))

for question_id in questions_list:
    feature_attributions = get_feature_attributions(question_id)
    top_k_features = get_top_k_features(feature_attributions, 20)
    run_intervention(question_id, top_k_features, good_features_list)
# %%

with open('./unlearning_output/good_features_list_v1.pkl', 'wb') as file:
    pickle.dump(good_features_list, file)
# %%
