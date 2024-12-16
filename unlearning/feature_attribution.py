
import sys
sys.path.append("../")

import torch
import random

from sae.sparse_autoencoder import load_saved_sae, SparseAutoencoder
from sae.metrics import model_store_from_sae
from unlearning.metrics import convert_wmdp_data_to_prompt, calculate_metrics_side_effects, calculate_metrics_list, create_df_from_metrics
from unlearning.tool import UnlearningConfig, SAEUnlearningTool, MCQ_ActivationStoreAnalysis

from huggingface_hub import hf_hub_download
from datasets import load_dataset
import numpy as np
import pandas as pd
import itertools
from transformer_lens import utils

from jaxtyping import Float
from torch import Tensor
import einops

import plotly.express as px
import pickle
import gc


def calculate_cache(model, prompt, answer, hook_point='blocks.9.hook_resid_pre'):
    """
    Takes in a multiple choice prompt and answer, computes forward and backward
    pass and returns cache based on logit diff
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    logits = model(tokens, return_type="logits")
    answer_strings = [" A", " B", " C", " D"]
    answer_strings = ["A", "B", "C", "D"]
    answer_tokens = model.to_tokens(answer_strings, prepend_bos=False).flatten()

    clear_contexts = False
    reset_hooks_end = True

    names_filter = [hook_point, hook_point + '_grad']
    
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter=names_filter, incl_bwd=True, device=None, remove_batch_dim=False
    )
    
    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        logits = model(tokens, return_type="logits")
        
        final_logits = logits[0, -1, answer_tokens]
        
        wrong_answers = torch.tensor([x for x in [0, 1, 2, 3] if x != answer]).to("cuda")
        
        # logit_diff = final_logits[answer] - final_logits[answer - 1]
        logit_diff = final_logits[answer] - final_logits[wrong_answers].mean()
        # logit_diff = final_logits[answer] - final_logits[wrong_answers[0]]
        # logit_diff = final_logits[answer] - logits[0, -1, 235305]

        logit_diff.backward()
        
        gc.collect()
        torch.cuda.empty_cache()

    model.reset_hooks()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return cache_dict


def find_topk_features_given_prompt(model,
                                    prompt,
                                    question,
                                    choices,
                                    answer,
                                    sae,
                                    hook_point,
                                    # ipos_min=15,
                                    # ipos_max=None,
                                    tpok_per_position=100,
                                    remove_newline_positions=True):
    
    """
    Get the topk feature attributions for given model and answer
    """
    # remove <bos> if it exists
    bos_token = model.tokenizer.bos_token
    if prompt.startswith(bos_token):
        prompt = prompt[len(bos_token):]

    cache_dict = calculate_cache(model, prompt, answer, hook_point)

    gc.collect()
    torch.cuda.empty_cache()
    
    len_context = cache_dict[hook_point].shape[1]

    question_len = model.to_tokens(question, prepend_bos=False).shape[-1] + 1
    inst_len = 15

    answer_lengths = [0] + [model.to_tokens(x, prepend_bos=False).shape[-1] + 3 for x in choices]
    cumulative_answer_lengths = np.cumsum(answer_lengths)

    correct_answer_start = inst_len + question_len + cumulative_answer_lengths[answer]
    correct_answer_end = inst_len + question_len + cumulative_answer_lengths[answer + 1]    

    # print(correct_answer_start, correct_answer_end)

    # Get the positions of the prompt associated with the question, and the correct answer
    # Ignoring the first word of the question and the question mark and newline token
    question_positions = np.arange(inst_len + 1, inst_len + question_len - 2)
    
    # Ignoring the "B. " and the final word and newline token
    correct_answer_positions = np.arange(correct_answer_start + 2, correct_answer_end - 2)

    # Calculate all answer positions
    all_answer_positions = []
    for j in range(4):
        answer_start = inst_len + question_len + cumulative_answer_lengths[j]
        answer_end = inst_len + question_len + cumulative_answer_lengths[j + 1]
        answer_positions = np.arange(answer_start + 2, answer_end - 2)
        all_answer_positions.append(answer_positions)

    all_answer_positions = [item for sublist in all_answer_positions for item in sublist]
           
    
    positions = np.concatenate((question_positions, all_answer_positions))
    # positions = np.arange(15, len(model.to_tokens(prompt)[0].cpu().numpy()) - 20)
    # print(positions)
        
    # positions = [i for i in positions if i not in remove_positions]

    gc.collect()
    torch.cuda.empty_cache()
    
    
    residual_activations = cache_dict[hook_point][0]
    
    if isinstance(sae, SparseAutoencoder):
        all_feature_activations, _ = sae(residual_activations)
    else:
        all_feature_activations = sae.encode(residual_activations)
            
    inds_list = []
    vals_list = []

    descending = False

    feature_attributions = []
    
    gc.collect()
    torch.cuda.empty_cache()
        
    # adjust for the GEMMA instruction format
    if '<start_of_turn>user' in prompt:
        positions = [pos + 3 for pos in positions]

    # Loops through every position
    for pos in positions:
        
        logit_diff_grad = cache_dict[hook_point + '_grad'][0, pos]
        
        with torch.no_grad():

            feature_activations = all_feature_activations[pos]
            scaled_features = einops.einsum(feature_activations, sae.W_dec, "feature, feature d_model -> feature d_model")
            feature_attribution = einops.einsum(scaled_features, logit_diff_grad, "feature d_model, d_model -> feature")
            feature_attributions.append(feature_attribution)
            
            vals, inds = feature_attribution.sort(descending=descending)

            vals_list.append(vals[:tpok_per_position])
            inds_list.append(inds[:tpok_per_position])
    
    gc.collect()
    torch.cuda.empty_cache()
        
    # Stack each position
    vals_subset = torch.vstack(vals_list)
    inds_subset = torch.vstack(inds_list)

    feature_attributions = torch.vstack(feature_attributions)
    
    # Get the indices corresponding to the sorted values in vals_subset
    v, i = vals_subset.flatten().sort(descending=descending)

    irow = torch.tensor([x // vals_subset.shape[1] for x in i])
    icol = torch.tensor([x % vals_subset.shape[1] for x in i])

    topk_features = torch.tensor([inds_subset[i, j] for i, j in zip(irow, icol)])
    topk_feature_attributions = torch.tensor([vals_subset[i, j] for i, j in zip(irow, icol)])
    
    indx = np.unique(topk_features.numpy(), return_index=True)[1]
    topk_features_unique = topk_features[sorted(indx)]
    
    irow = torch.tensor([positions[x] for x in irow])
        
    # return topk_features_unique
    return topk_features_unique #, feature_attributions, topk_features, all_feature_activations, logit_diff_grad, topk_feature_attributions
    
from unlearning.metrics import modify_and_calculate_metrics
from tqdm import tqdm

def test_topk_features(model,
                       sae,
                       question_id,
                       features_to_ablate,
                       known_good_features=[],
                       multiplier=20,
                       thres_correct_ans_prob=0.4,
                       permutations=[[0, 1, 2, 3]]):
    """
    Test a bunch of features
    """
    
    intervention_results = []

    # Ablate each feature one by one
    features_to_ablate = [i for i in features_to_ablate if i not in known_good_features]
    
    for feature in tqdm(features_to_ablate):

        # print("Testing feature #", feature.item())
        
        ablate_params = {
            'features_to_ablate': [feature],
            'multiplier': multiplier,
            'intervention_method': 'clamp_feature_activation',
        }

        # print(feature)

        metrics = modify_and_calculate_metrics(model,
                                 sae,
                                 dataset_names=['wmdp-bio'],
                                 metric_params={'wmdp-bio': {'question_subset': [question_id], 'permutations': permutations, 'verbose': False}},
                                 n_batch_loss_added=2,
                                 activation_store=None,
                                 **ablate_params)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        #     intervened_correct_ans_prob = metrics['wmdp-bio']['predicted_probs_of_correct_answers'].item()
    
        #     if intervened_correct_ans_prob < thres_correct_ans_prob:
        #         good_features.append(feature.item())
    
        # return good_features
        
        #     ablate_params = {
        #         'features_to_ablate': [feature],
        #         'multiplier': multiplier,
        #         'intervention_method': 'scale_feature_activation',
        #         'question_subset_file': None,
        #         'question_subset': [question_id],
        #     }
    
        #     metrics = ul_tool.calculate_metrics(**ablate_params)
        intervention_results.append(metrics)
        
    # prob_correct = [metrics['wmdp-bio']['predicted_probs_of_correct_answers'].mean().item() for metrics in intervention_results]
    prob_correct = [metrics['wmdp-bio']['predicted_probs_of_correct_answers'].min().item() for metrics in intervention_results]
    features_to_ablate = [j.item() for j in features_to_ablate]
    feature_ids_to_probs = dict(zip(features_to_ablate, prob_correct))

    if permutations is None:
        permutations = [[0, 1, 2, 3]]
    partially_unlearned = np.array([metrics['wmdp-bio']['is_correct'].sum() < len(permutations) for metrics in intervention_results])
    good_features = [j for j, unlearned in zip(features_to_ablate, partially_unlearned) if unlearned]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return intervention_results, feature_ids_to_probs, good_features
      
      
def get_topk_features_by_attribution(model, sae, question_ids, prompts, choices_list, answers, questions, topk_per_prompt=20):
    feature_per_prompt = {}
    
    known_good_features = []
        
    for j, question_id in enumerate(question_ids):
    
        question_id = int(question_id)
        # print(f"Question ID: {question_id}, {j + 1}/{len(question_ids)}")
        
        prompt = prompts[question_id]
        choices = choices_list[question_id]
        answer = answers[question_id]
        question = questions[question_id]
    
        topk_features_unique = find_topk_features_given_prompt(model,
                                                               prompt,
                                                               question,
                                                               choices,
                                                               answer,
                                                               sae,
                                                               hook_point=sae.cfg.hook_point)
    
        intervention_results, feature_ids_to_probs, good_features = test_topk_features(model,
                                                                                       sae,
                                                                                       question_id,
                                                                                       topk_features_unique[:topk_per_prompt],
                                                                                       known_good_features=known_good_features,
                                                                                       multiplier=30)
        
    
        feature_per_prompt[question_id] = good_features
        
        known_good_features = list(set([item for sublist in feature_per_prompt.values() for item in sublist]))

    return feature_per_prompt, known_good_features
    


def get_features_without_side_effects(model,
                                      sae,
                                      known_good_features, 
                                      thresh=0,
                                      target_metric='correct',
                                      split='all',
                                      multiplier=30,
                                      intervention_method='clamp_feature_activation'):
    
    # Calculate metrics
    
    main_ablate_params = {
                          'multiplier': multiplier,
                          'intervention_method': intervention_method,
                         }
    
    
    sweep = {
             'features_to_ablate': np.array(known_good_features),
            }
    
    
    dataset_names = ['high_school_us_history', 'college_computer_science', 'high_school_geography', 'human_aging']
    
    metrics_list = calculate_metrics_side_effects(model,
                                          sae,
                                          main_ablate_params,
                                          sweep,
                                          dataset_names=dataset_names,
                                          thresh=thresh,
                                          target_metric=target_metric,
                                          split=split)

    feature_ids_zero_side_effect = [x['ablate_params']['features_to_ablate'] for x in metrics_list]

    return metrics_list, feature_ids_zero_side_effect
    
    


def sort_by_loss_added(model,
                       sae,
                       feature_ids_zero_side_effect,
                       question_ids,
                       activation_store,
                       multiplier=20,
                       intervention_method='clamp_feature_activation',
                       n_batch_loss_added=20,
                       split='all',
                       verbose=False,
                      ):

    # Calculate metrics
    
    main_ablate_params = {
                          'multiplier': multiplier,
                          'intervention_method': intervention_method,
                         }
    
    sweep = {
             'features_to_ablate': feature_ids_zero_side_effect,
            }
    
    metric_params = {'wmdp-bio': 
                     {
                           'question_subset': question_ids,
                           'permutations': None,
                           'verbose': verbose,
                       }
                     }
    
    dataset_names = ['loss_added', 'wmdp-bio']
    
    metrics_list_zero_side_effect = calculate_metrics_list(model,
                                          sae,
                                          main_ablate_params,
                                          sweep,
                                          dataset_names=dataset_names,
                                          metric_params=metric_params,
                                          include_baseline_metrics=False,
                                          n_batch_loss_added=n_batch_loss_added,
                                          activation_store=activation_store,
                                          split=split,
                                          verbose=verbose)

    df_zero_side_effect = create_df_from_metrics(metrics_list_zero_side_effect)

    isorted = df_zero_side_effect.query("`wmdp-bio` < 1").sort_values("loss_added").index.values
    feature_ids_zero_side_effect_sorted = np.array(feature_ids_zero_side_effect)[isorted]

    return df_zero_side_effect, feature_ids_zero_side_effect_sorted

    
    
    
def add_features_sorted_by_loss(model,
                               sae,
                               features_ids_zero_side_effect_sorted,
                               istart=0,
                               multiplier=30,
                               intervention_method='clamp_feature_activation',
                               n_batch_loss_added=20,
                               ):

    # Calculate metrics
    
    main_ablate_params = {
                          'multiplier': multiplier,
                          'intervention_method': intervention_method,
                         }
    
    
    sweep = {
             'features_to_ablate': [feature_ids_zero_side_effect_sorted[:i+1] for i in range(istart, len(feature_ids_zero_side_effect_sorted))],
            }
    
    metric_params = {'wmdp-bio': 
                     {
                           'target_metric': 'correct',
                           'permutations': None,
                           'verbose': False,
                       }
                     }
    
    dataset_names = ['loss_added', 'wmdp-bio']
    
    metrics_list_best_sorted = calculate_metrics_list(model,
                                          sae,
                                          main_ablate_params,
                                          sweep,
                                          dataset_names=dataset_names,
                                          metric_params=metric_params,
                                          include_baseline_metrics=False,
                                          n_batch_loss_added=n_batch_loss_added,
                                          activation_store=activation_store,
                                          split='test')
    
    







    
    