# %%
from unlearning.data_utils import save_target_question_ids, save_train_test_all
import transformer_lens
import torch

# model = transformer_lens.HookedTransformer.from_pretrained("gemma-2b-it")
model = transformer_lens.HookedTransformer.from_pretrained("gemma-2-2b-it")
# model = transformer_lens.HookedTransformer.from_pretrained_no_processing("google/gemma-2-9b-it", dtype=torch.bfloat16)
print(model.cfg.model_name)

for dataset_name in ['wmdp-bio', 'high_school_us_history', 'college_computer_science', 'high_school_geography', 'human_aging', 'college_biology']:
    save_target_question_ids(model, dataset_name) 
    save_train_test_all(dataset_name, model.cfg.model_name)