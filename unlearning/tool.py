import json
import pathlib
import os
import torch

from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer, utils
from huggingface_hub import hf_hub_download
from functools import partial
import gc
from datasets import load_dataset
from collections.abc import Iterable

from rich.table import Table
from rich.console import Console
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from jaxtyping import Float

from unlearning.metrics import convert_wmdp_data_to_prompt, calculate_MCQ_metrics, get_tokens_from_dataset
from unlearning.intervention import anthropic_remove_resid_SAE_features, remove_resid_SAE_features, anthropic_clamp_resid_SAE_features
from unlearning.var import REPO_ID, SAE_MAPPING

from sae.sparse_autoencoder import load_saved_sae
from sae.metrics import model_store_from_sae, compute_metrics_post
from sae.activation_store import ActivationStoreAnalysis
from sae.utils import log
from sae.config import Config
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.nn.functional as F


import torch.nn as nn
from types import SimpleNamespace


def get_basic_gemma_2b_it_layer9_act_store(model):
    # resid pre 9
    REPO_ID = "eoinf/unlearning_saes"
    FILENAME = "jolly-dream-40/sparse_autoencoder_gemma-2b-it_blocks.9.hook_resid_pre_s16384_127995904.pt"

    filename = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    sae = load_saved_sae(filename)

    activation_store = ActivationStoreAnalysis(sae.cfg, model)
    
    return activation_store



class MCQ_ActivationStoreAnalysis(ActivationStoreAnalysis):
   

    def __init__(
        self,
        cfg,
        model: HookedTransformer,
        dataset = 'wmdp-bio',
        dataset_args = {},
    ):
        """
        Subclass of ActivationStoreAnalysis for multiple choice questions (MCQ)
        """
        super().__init__(cfg, model, create_dataloader=False, shuffle_buffer=False)

        dataset_path_mapping = {
            'wmdp-bio': ("cais/wmdp", "wmdp-bio"),
            'mmlu-bio': ("cais/mmlu", "college_biology"),
            'mmlu-high_school_us_history': ("cais/mmlu", "high_school_us_history"),
        }
        assert dataset in dataset_path_mapping.keys(), f"Dataset {dataset} not found in {dataset_path_mapping.keys()}"
        dataset_path = dataset_path_mapping[dataset]
        
        self.tokenized_dataset = get_tokens_from_dataset(model, dataset_path=dataset_path, **dataset_args)
        self.start_row = 0

        # log("buffer")
        # self.storage_buffer, self.storage_buffer_tokens = self.generate_buffer(self.cfg.n_batches_in_store_buffer // 2)
        log("dataloader")
        self.dataloader, self.dataloader_tokens = self.get_data_loader()


    def get_batch_tokenized_data(self):
        """
        Returns a batch of tokenized data
        """
        # Defines useful variables
        store_batch_size = self.cfg.store_batch_size
        device = self.cfg.device
        n_rows = self.tokenized_dataset.shape[0]

        end_row = (self.start_row + self.cfg.store_batch_size) 
        
        if end_row < n_rows:
            batch_tokens = self.tokenized_dataset[self.start_row:end_row]
            self.start_row = end_row % n_rows
        else:
            batch_end = self.tokenized_dataset[self.start_row:]
            new_start_row = self.cfg.store_batch_size - batch_end.shape[0]
            batch_start = self.tokenized_dataset[:new_start_row]
            batch_tokens = torch.vstack((batch_end, batch_start))
            self.start_row = new_start_row

        return batch_tokens

    def get_data_loader(
        self,
    ) -> DataLoader:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        # 1. # create new buffer by mixing stored and new buffer
        buffer, buffer_tokens = self.generate_buffer(self.cfg.n_batches_in_store_buffer)
        
        # mixing_buffer = torch.cat(
        #     [buffer, self.storage_buffer]
        # )
        
        # mixing_buffer_tokens = torch.cat(
        #     [buffer_tokens, self.storage_buffer_tokens]
        # )

        dataloader = iter(
            DataLoader(
                buffer,
                batch_size=self.cfg.train_batch_size,
            )
        )

        dataloader_tokens = iter(
            DataLoader(
                buffer_tokens,
                batch_size=int(self.cfg.train_batch_size),
            )
        )

        return dataloader, dataloader_tokens
        

@dataclass()
class UnlearningConfig():
    
    sae_key: str = "gemma_2b_it_resid_pre_9"
    base_dataset: str = "skylion007/openwebtext"
    unlearn_dataset: str | None = None
    output_dir: str = './unlearning_output/'

    # unlearning_activation_store: str = 'wmdp-bio'
    unlearn_activation_store: MCQ_ActivationStoreAnalysis | None = None
    unlearning_metric: str = 'wmdp-bio'
    control_metric: str | list = field(default_factory=lambda: ['high_school_us_history', 'college_computer_science', 'high_school_geography', 'human_aging', 'college_biology'])
    filter_correct_questions: bool = True


class SAEUnlearningTool:
    """
    A postprocessing tool for running unlearning experiments using a pretrained SAE.
    """

    def __init__(self, cfg: UnlearningConfig):
        """
        Initializes an unlearning tool that contains:
            1. A language model and pretrained SAE
            2. A "base" dataset that contains broad text content (e.g., openwebtext)
            3. An "unlearn" dataset that contains text targeted at the topic that should be unlearned.
        """
        self.cfg = cfg

    def setup(self,
              create_base_act_store=True,
              create_unlearn_act_store=True,
              model=None,
              custom_sae=None):

        # Load SAE
        if custom_sae is not None:
            self.sae = custom_sae
        else:
            filename = hf_hub_download(repo_id=REPO_ID, filename=SAE_MAPPING[self.cfg.sae_key])
            self.sae = load_saved_sae(filename)

        # Load language model
        if model is None:
            self.model = model_store_from_sae(self.sae)
        else:
            self.model = model

        # Load base activation store
        self.sae.cfg.use_cached_activations = False
        self.sae.cfg.dataset = self.cfg.base_dataset
        if create_base_act_store:
            self.base_activation_store = ActivationStoreAnalysis(self.sae.cfg, self.model)

        # Load unlearn activation store
        if create_unlearn_act_store:
            self.unlearn_activation_store = self.cfg.unlearn_activation_store
        
        self.console = Console()

        # Load dataset for metrics
        self.load_wmdp_bio_dataset()
        
        # Namespace upkeep
        self.base_metrics_with_text = None
        self.unlearn_metrics_with_text = None
        self.merged_feature_dfs = None
        self.feature_dfs = dict()

        self.out_dir = pathlib.Path(self.cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        
    def load_wmdp_bio_dataset(self):
        self.dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    
    def get_metrics_with_text(
        self, 
        save_learned_activations: bool = True, 
        n_batches: int = 43, # 100, 
        len_prefix: int = 5
        ):
        """Calls the metrics.compute_metrics_post() for both datasets and stores the results."""
        kwargs = {
            'save_learned_activations' : save_learned_activations,
            'n_batches' : n_batches,
            'len_prefix' : len_prefix
        }
        self.base_metrics_with_text = compute_metrics_post(
            self.sae,
            self.base_activation_store,
            self.model,
            **kwargs
        )
        self.unlearn_metrics_with_text = compute_metrics_post(
            self.sae,
            self.unlearn_activation_store,
            self.model,
            **kwargs
        )

    def get_top_features_on_token(
            self, 
            prompt: str, 
            object: str, 
            k: int = 30,
            hide_table: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a prompt and an object string contained in the prompt, this function
        returns up to k features with the highest SAE activations on the last token in
        the prompt making up the object string.

        Returns a tuple of two tensors: the first tensor contains the SAE feature indices
        while the second tuple contains the corresponding activations.
        """
        #forward pass of model
        tokens = self.model.to_tokens(prompt)
        important_hooks = [self.sae.cfg.hook_point,]
        _, cache = self.model.run_with_cache(tokens, names_filter=important_hooks)
        activations = cache[important_hooks[0]]

        #SAE forward pass
        feature_activations, _ = self.sae(activations)

        #Get top-k activations on desired token.
        obj_token = self.model.to_tokens(object)[0,-1].item()
        top_k = torch.topk(feature_activations, k, dim=-1)
        where = torch.where(tokens == obj_token)
        act_val = top_k.values[where[0], where[1]].flatten()
        act_feat = top_k.indices[where[0], where[1]].flatten()

        if not hide_table:
            # Create a table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("act_feat", style="dim", width=12)
            table.add_column("act_val", style="dim", width=12)

            # Add rows to the table
            for left, right in zip(act_feat, act_val):
                table.add_row(str(left.item()), str(right.item()))

            self.console.print(table)
        return act_feat, act_val

    def get_top_features_on_substrings(
            self, 
            prompt: str, 
            substrings: list[str], 
            k: int = 30,
            hide_table: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a prompt and a list of object strings contained in the prompt, this function
        returns up to k features with the highest SAE activations on the object string s.

        Returns a tuple of two tensors: the first tensor contains the SAE feature indices
        while the second tuple contains the corresponding activations.
        """
        #forward pass of model
        tokens = self.model.to_tokens(prompt)
        important_hooks = [self.sae.cfg.hook_point,]
        _, cache = self.model.run_with_cache(tokens, names_filter=important_hooks)
        activations = cache[important_hooks[0]]

        #SAE forward pass
        feature_activations, _ = self.sae(activations)

        # Get the token indices for each substring.
        token_indices = []
        for substr in substrings:
            substr = self.model.to_tokens(substr, prepend_bos=False)[0]
            for j in range(tokens.shape[1]):
                if tokens[0,j].item() == substr[0].item(): 
                    if torch.allclose(tokens[0,j:j+substr.shape[0]], substr): 
                        token_indices.extend(np.arange(j, j+substr.shape[0]).tolist())

        # feature_activations has shape [batch, pos, d_sae] -- batch is len() == 1.
        feature_activations = feature_activations[0, token_indices, :]
        top_k = torch.topk(feature_activations, k, dim=-1)
        act_val = top_k.values.flatten()
        act_feat = top_k.indices.flatten()
        
        
        # Sort act_val in descending order and get the sorted indices
        sorted_indices = torch.argsort(act_val, descending=True)
        sorted_act_val = act_val[sorted_indices]
        sorted_act_feat = act_feat[sorted_indices]

        # Use a dictionary to keep the highest activation for each feature
        # torch.unique is a bit tricky and hard to understand here.
        unique_feat = {}
        for val, feat in zip(sorted_act_val, sorted_act_feat):
            if feat.item() not in unique_feat:
                unique_feat[feat.item()] = val

        # Extract the filtered values
        filtered_act_feat = torch.tensor(list(unique_feat.keys()))
        filtered_act_val = torch.tensor(list(unique_feat.values()))

        #re-do topk to actually get down to correct top-k
        top_k = torch.topk(filtered_act_val, k)
        act_val = top_k.values
        act_feat = filtered_act_feat[top_k.indices]

        if not hide_table:
            # Create a table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("act_feat", style="dim", width=12)
            table.add_column("act_val", style="dim", width=12)

            # Add rows to the table
            for left, right in zip(act_feat, act_val):
                table.add_row(str(left.item()), str(right.item()))

            self.console.print(table)
        return act_feat, act_val

    def generate_feature_activation_df(self, feature_ids: list[int]):
        """
        Given a list of feature indices, this function generates dataframes for each feature
        showing the top activations for that feature in the base and unlearn datasets.
        These dataframes are stored in the self.feature_dfs dict.
        """

        if self.base_metrics_with_text is None and self.unlearn_metrics_with_text is None:
            raise ValueError("Run get_metrics_with_text() first to get activation metrics")
        
        for feature_id in feature_ids:
            self.feature_dfs[feature_id] = dict()
            metrics_with_texts = [self.base_metrics_with_text, self.unlearn_metrics_with_text]
            labels = ['base', 'unlearn']
            for metrics, label in zip(metrics_with_texts, labels):
                metrics['token_df']["activation"] = utils.to_numpy(metrics['learned_activations'][:, feature_id])
                df = metrics['token_df'][['str_tokens','prefix', 'suffix',  'context', 'activation']]
                self.feature_dfs[feature_id][label] = df.copy()
    
    def merge_feature_dfs(self):
        """ 
        This function saves images of the feature activation dataframes.

        This function requires the dataframe_image package to be installed; to install it, run
            pip install -q dataframe_image
        
        """
        if len(self.feature_dfs) == 0:
            raise ValueError("Run generate_feature_activation_df() first to generate dataframes")

        df_list = []
        for feature_id in tqdm(self.feature_dfs.keys()):
            these_dfs = []
            for label, df in self.feature_dfs[feature_id].items():
                df = df.copy()
                df['Dataset'] = label
                these_dfs.append(df)
            merged_df = pd.concat(these_dfs, ignore_index=True)
            merged_df["feature_id"] = feature_id
            merged_df = merged_df.sort_values("activation", ascending=False).head(1000) #only keep top 1k activations
            df_list.append(merged_df)
        self.merged_feature_dfs = pd.concat(df_list, ignore_index=True)
        # display(self.merged_feature_dfs.head())
    
    def print_feature_df_head(self, feature_id, num_rows=20, extra_title=""):
        if isinstance(feature_id, torch.Tensor):
            feature_id = feature_id.item()
        partial_df = self.merged_feature_dfs[self.merged_feature_dfs["feature_id"] == feature_id]
        title = f"# Feature ID: {feature_id}"
        if extra_title:
            title += f"; {extra_title}"
        display(Markdown(title))
        display(partial_df.head(num_rows).style.background_gradient("coolwarm"))
    
    def reset_feature_dfs(self):
        del self.feature_dfs
        self.feature_dfs = dict()
        del self.merged_feature_dfs
        self.merged_feature_dfs = None
    
    def generate_text_with_ablated_features(
            self, 
            prompt: str, 
            features_to_ablate: list, 
            multiplier: float = 1.0,
            generate_kwargs=None,
            print_outputs=True,
            test_baseline=True,
            use_anthropic_method=True,
        ):
        """
        This function takes a prompt and a list of feature indices to ablate.
        It first generates a baseline sample of text *without* any features ablated,
        then it generates text *with* the specified features ablated.
        """
        self.model.reset_hooks()
        
        if generate_kwargs is None:
            generate_kwargs = {
                'max_new_tokens': 100,
                'do_sample': True,
                'temperature': 0
            }

        with torch.no_grad():
            if test_baseline:
                default_output = self.model.generate(prompt, **generate_kwargs).split(prompt)[-1]
            else:
                default_output = "Harry"
                
        if print_outputs:
            print("#### Default ####")
            self.console.print(f'[bold yellow]{prompt}[/bold yellow]{default_output}')

        if use_anthropic_method:
            ablation_method = anthropic_remove_resid_SAE_features
        else:
            ablation_method = remove_resid_SAE_features
        ablate_hook_func = partial(
            ablation_method, 
            sae=self.sae, 
            features_to_ablate=features_to_ablate,
            multiplier=multiplier
            )
        self.model.add_hook(self.sae.cfg.hook_point, ablate_hook_func)

        with torch.no_grad():
            modified_output = self.model.generate(prompt, **generate_kwargs).split(prompt)[-1]
        
        if print_outputs:
            print("#### Intervened ####")
            self.console.print(f'[bold yellow]{prompt}[/bold yellow]{modified_output}')

        self.model.reset_hooks()
        
        return default_output, modified_output
    
    def get_unlearning_metrics(
            self, 
            prompt: str, 
            next_token_ids: list[int],
            features_to_ablate: list[int],
            multiplier: float = 1.0,
            use_anthropic_method: bool = False
        ):
        """
        This function returns logit-based metrics for the model's performance on a prompt
        with and without specified features ablated.

        Metrics are only calculated for the next token at the end of the prompt, and only
        for the token ids specified in next_token_ids.

        Current metrics:
            delta_logits -- The change in logit value for the next tokens
            delta_prob -- The change in probability for the next tokens
            fractional_prob_change -- the change in probability for the next tokens, 
                            normalized by their values in the baseline forward pass.
        """
        metrics = dict()

        self.model.reset_hooks()

        baseline_logits = self.model(prompt)[0,-1,:] #get logits for next token
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_logits = baseline_logits[next_token_ids]
        baseline_probs = baseline_probs[next_token_ids]

        if use_anthropic_method:
            ablation_method = anthropic_remove_resid_SAE_features
        else:
            ablation_method = remove_resid_SAE_features
        ablate_hook_func = partial(
            ablation_method, 
            sae=self.sae, 
            features_to_ablate=features_to_ablate,
            multiplier=multiplier
            )
        self.model.add_hook(self.sae.cfg.hook_point, ablate_hook_func)
        unlearn_logits = self.model(prompt)[0,-1,:]
        self.model.reset_hooks()
        unlearn_probs = torch.softmax(unlearn_logits, dim=-1)
        unlearn_logits = unlearn_logits[next_token_ids]
        unlearn_probs = unlearn_probs[next_token_ids]

        metrics['delta_logits'] = unlearn_logits - baseline_logits
        metrics['delta_prob'] = unlearn_probs - baseline_probs
        metrics['fractional_prob_change'] = metrics['delta_prob'] / baseline_probs

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[key] = value.item()
                else:
                    metrics[key] = value.cpu()
        
        return metrics

    # intervention_method: str = 'scale_feature_activation' 
    # custom_hook_point: Optional[str] = None

    def modify_model(self, **kwargs):

        default_modification_kwargs = {
            'multiplier': 1.0,
            'intervention_method': 'scale_feature_activation',
            'custom_hook_point': None,
        }
        
        self.model.reset_hooks()
        
        # Calculate modified stats
        if kwargs['intervention_method'] == "scale_feature_activation":
            ablation_method = anthropic_remove_resid_SAE_features
        elif kwargs['intervention_method'] == "remove_from_residual_stream":
            ablation_method = remove_resid_SAE_features
        elif kwargs['intervention_method'] == "clamp_feature_activation":
            ablation_method = anthropic_clamp_resid_SAE_features
            
        ablate_hook_func = partial(
            ablation_method, 
            sae=self.sae, 
            features_to_ablate=kwargs['features_to_ablate'],
            multiplier=kwargs['multiplier']
            )
        
        if 'custom_hook_point' not in kwargs or kwargs['custom_hook_point'] is None:
            hook_point = self.sae.cfg.hook_point
        else:
            hook_point = kwargs['custom_hook_point']
        
        self.model.add_hook(hook_point, ablate_hook_func)

    def get_baseline_metrics(self, recompute=False):
        """
        Compute the baseline metrics or retrieve if pre-computed and saved
        """
        baseline_metrics_file = '../data/' + self.cfg.unlearning_metric + '.json'

        if not recompute and os.path.exists(baseline_metrics_file):
            
            # Load the json
            with open(baseline_metrics_file, "r") as f:
                baseline_metrics = json.load(f)

            # Convert lists to arrays for ease of use
            for key, value in baseline_metrics.items():
                if isinstance(value, list):
                    baseline_metrics[key] = np.array(value)
                    
            return baseline_metrics

        else:
            baseline_metrics = calculate_MCQ_metrics(self.model, target_metric=self.cfg.unlearning_metric)
            
            metrics = baseline_metrics.copy()
            # Convert lists to arrays for ease of use
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()                         

            with open(baseline_metrics_file, "w") as f:
                json.dump(metrics, f)

            return baseline_metrics
            
        
    def calculate_control_metrics(self, **kwargs):
        
        self.model.reset_hooks()
        self.modify_model(**kwargs)
        
        if isinstance(self.cfg.control_metric, str):
            self.cfg.control_metric = [self.cfg.control_metric]

        control_metrics = {}
        for metric in self.cfg.control_metric:
            full_metric = f'mmlu_{metric}_{self.model.cfg.model_name}_correct_not_correct_wo_question_prompt'
            control_metrics[metric] = calculate_MCQ_metrics(self.model, target_metric=full_metric, **kwargs)
                
        self.model.reset_hooks()
        return control_metrics
    
    
    def calculate_metrics(self, **kwargs):

        metrics = {}

        self.model.reset_hooks()
        
        # Calculate baseline metrics
        baseline_metrics = self.get_baseline_metrics()
        metrics['baseline_metrics'] = baseline_metrics

        # Calculate modified metrics
        self.modify_model(**kwargs)
        print("calling MCQ metrics")
        modified_metrics = calculate_MCQ_metrics(self.model, target_metric=self.cfg.unlearning_metric, **kwargs)
        
        metrics['modified_metrics'] = modified_metrics    
        self.model.reset_hooks()
        
        
        


        # metrics['multiplier'] = multiplier
        # metrics['features_to_ablate'] = features_to_ablate
        
        # metrics['model_probs'] = model_probs
        # metrics['baseline_probs'] = baseline_probs
        
        # metrics['mean_prob_ratio'] = np.mean(model_probs/baseline_probs)
        # metrics['median_prob_ratio'] = np.median(model_probs/baseline_probs)
        # metrics['fractional_prob_change'] = np.mean((model_probs - baseline_probs)/baseline_probs)
        
        # # Compute the percentage that the modified model got correct
        # metrics['percent_correct'] = np.mean(model_correct)/np.mean(baseline_correct)
        
        # # Compute loss added
        # loss_added = self.get_loss_added(features_to_ablate=features_to_ablate,
        #                         multiplier=multiplier,
        #                         n_batch=n_batch,
        #                         custom_hook_point=custom_hook_point,
        #                         use_anthropic_method=use_anthropic_method)

        # metrics['loss_added'] = loss_added
        
        return metrics

    def compute_loss_added(self, n_batch=2, verbose=True, **kwargs):
        # Gets loss
        
        with torch.no_grad():
            loss_diffs = []
            
            for _ in tqdm(range(n_batch), disable=not verbose):
                
                tokens = self.base_activation_store.get_batch_tokenized_data()
                
                # Compute baseline loss
                self.model.reset_hooks()
                baseline_loss = self.model(tokens, return_type="loss")
                    
                gc.collect()
                torch.cuda.empty_cache()
            
        
                # Calculate modified loss
                self.model.reset_hooks()
                self.modify_model(**kwargs)
                # modified_metrics = unlearning_metric_functions[self.cfg.unlearning_metric](self.model, **kwargs)    
                # self.model.reset_hooks()
                # if use_anthropic_method:
                #     ablation_method = anthropic_remove_resid_SAE_features
                # else:
                #     ablation_method = remove_resid_SAE_features
                # ablate_hook_func = partial(
                #     ablation_method, 
                #     sae=self.sae, 
                #     features_to_ablate=features_to_ablate,
                #     multiplier=multiplier
                #     )
                # if custom_hook_point is None:
                #     hook_point = self.sae.cfg.hook_point
                # else:
                #     hook_point = custom_hook_point
                # self.model.add_hook(hook_point, ablate_hook_func)
                modified_loss = self.model(tokens, return_type="loss")
                
                gc.collect()
                torch.cuda.empty_cache()
                
                self.model.reset_hooks()
                
                loss_diff = modified_loss.item() - baseline_loss.item()
                loss_diffs.append(loss_diff)
            
            return np.mean(loss_diffs)
 
    def get_loss_added(
            self,
            features_to_ablate,
            multiplier,
            n_batch=1,
            custom_hook_point=None,
            use_anthropic_method: bool = False
        ):
        #TODO: Fix this so that we can use the same tokens for many different ablation experiments.
        
        with torch.no_grad():
            loss_diffs = []
            
            for _ in range(n_batch):
                
                tokens = self.base_activation_store.get_batch_tokenized_data()
                # # Compute baseline loss
                self.model.reset_hooks()
                baseline_loss = self.model(tokens, return_type="loss")
                    
                gc.collect()
                torch.cuda.empty_cache()
            
                self.model.reset_hooks()
                # Calculate modified loss
                if use_anthropic_method:
                    ablation_method = anthropic_remove_resid_SAE_features
                else:
                    ablation_method = remove_resid_SAE_features
                ablate_hook_func = partial(
                    ablation_method, 
                    sae=self.sae, 
                    features_to_ablate=features_to_ablate,
                    multiplier=multiplier
                    )
                if custom_hook_point is None:
                    hook_point = self.sae.cfg.hook_point
                else:
                    hook_point = custom_hook_point
                self.model.add_hook(hook_point, ablate_hook_func)
                modified_loss = self.model(tokens, return_type="loss")
                
                gc.collect()
                torch.cuda.empty_cache()
                
                self.model.reset_hooks()
                
                loss_diff = modified_loss.item() - baseline_loss.item()
                loss_diffs.append(loss_diff)
            
            return np.mean(loss_diffs)
    
    # def test_completion_with_dictionary(
    #     self,
    #     features_to_ablate,
    #     multiplier,
    #     n_completion_prompts=5,
    #     max_new_tokens=50,
    #     temperature=0.0,
    #     test_baseline=False,
    #     use_anthropic_method: bool = False
    # ):
    #     hp_dictionary_file = "../who_is_harry_potter/data/harry_potter_dictionary.json"
    #     with open(hp_dictionary_file) as f:
    #         hp_dict = json.load(f) 

    #     hp_dict = [x.lower() for x in hp_dict]
        
    #     generate_kwargs = {'max_new_tokens': max_new_tokens,
    #                       'do_sample': True,
    #                       'temperature': temperature}    
    
    #     default_output_list = []
    #     modified_output_list = []

    #     if n_completion_prompts == "all":
    #         completion_prompts = COMPLETION_PROMPTS
    #     else:
    #         completion_prompts = COMPLETION_PROMPTS[:n_completion_prompts]
        
    #     for prompt in completion_prompts:
            
    #         default_output, modified_output = self.generate_text_with_ablated_features(
    #             prompt, 
    #             features_to_ablate=features_to_ablate,
    #             multiplier=multiplier,
    #             generate_kwargs=generate_kwargs,
    #             print_outputs=False,
    #             test_baseline=test_baseline,
    #             use_anthropic_method=use_anthropic_method
    #         )
    #         default_output_list.append(default_output)
    #         modified_output_list.append(modified_output)   
        
    #     default_contains_hp_data_list = []
    #     modified_contains_hp_data_list = []
      
    #     for default_output, modified_output, prompt in zip(default_output_list, modified_output_list, completion_prompts):
    #         hp_dict_excluding_prompt = [x for x in hp_dict if x not in prompt.lower()]
    #         default_contains_hp_data = any(x in default_output.lower() for x in hp_dict_excluding_prompt)
    #         modified_contains_hp_data = any(x in modified_output.lower() for x in hp_dict_excluding_prompt)
            
    #         default_contains_hp_data_list.append(default_contains_hp_data)
    #         modified_contains_hp_data_list.append(modified_contains_hp_data)
        
    #     metrics = {}

    #     metrics['default_output'] = default_output_list
    #     metrics['modified_output'] = modified_output_list
        
    #     if test_baseline:
    #         metrics['default_contains_hp_data'] = default_contains_hp_data_list
    #     else:
    #         metrics['default_contains_hp_data'] = [1 for _ in default_contains_hp_data_list]

    #     metrics['modified_contains_hp_data'] = modified_contains_hp_data_list
        
    #     metrics['baseline_completion_familiarity'] = np.mean(metrics['default_contains_hp_data'])
    #     metrics['completion_familiarity'] = np.mean(metrics['modified_contains_hp_data'])
       
    #     metrics['prompt'] = completion_prompts
        
    #     return metrics
    
    
def get_hf_model(model_name: str, tokenizer_model_name: str = None):
    "get huggingface model, setup tokenizer and important cfg"
    if not tokenizer_model_name:
        tokenizer_model_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    model.tokenizer = tokenizer
    model.cfg = SimpleNamespace()
    model.cfg.model_name = tokenizer_model_name.split('/')[-1]
    
    return model