import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
import pandas as pd
from tqdm import tqdm
import time
import argparse
import json
import pickle
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils.utils import set_seed
import torch
import copy
import re
parser = argparse.ArgumentParser()

# Required parameters

parser.add_argument("--dataset", 
                    default="beauty", 
                    choices=["yelp", "fashion", "beauty", "steam"],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument('--filter_user_id',
                    type=int,
                    default=2678,
                    help="for testing prompt performance, filtered one user's records")
parser.add_argument("--is_filter_user_id",
                    default=False,
                    help="is filter_user_id")
parser.add_argument('--terminate',
                    type=int,
                    default=1,
                    help="for filter batch_id")
parser.add_argument('--span_len',
                    type=int,
                    default=7,
                    help="stage size of sequence")
parser.add_argument('--max_sub_batch_size',
                    type=int,
                    default=12,
                    help="batch_size")
parser.add_argument("--is_saved_emb",
                    default=True,
                    help="is saved emb")
parser.add_argument("--is_saved_text",
                    default=True,
                    help="is saved LLM response")
parser.add_argument('--gpu_id',
                    type=int,
                    default=6,
                    help="gpu id")
parser.add_argument('--deleted',
                    action='store_true',
                    default=False,
                    help="Whether LLM filters items")
parser.add_argument('--router_decision',
                    action='store_true',
                    default=False,
                    help="Whether LLM filters items")
args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

set_seed(args.seed) # fix the random seed
ENVIRONMENT_PROMPT_TEMPLATE = f"On the Amazon e-commerce platform, the user has interaction records exclusively from the one categories: {args.dataset}. The following is the historical interaction record of the user at the current stage.\n"
if args.dataset == "yelp":
    ENVIRONMENT_PROMPT_TEMPLATE = f"The following is the historical interaction record of the user at the current stage.\n"
if args.dataset == "steam":
    ENVIRONMENT_PROMPT_TEMPLATE = f"On the steam game platform, the user has review records. The following is the historical interaction record of the user at the current stage.\n"
 
USER_PREVIOUS_INTERESTS_TEMPLATE = "The user's previous interests: {} \n"

OUTPUT_NO_INTEREST_PROMPT_TEMPLATE = """Please follow the steps below to infer the user's preferences, personality and shopping habits.\n
Step 1: Please infer the user's current preferences at this stage, focusing primarily on the types of items the user might like, which can be used for recommending items in the future. \n 
Step 2: When summarizing users' interests, consider the granularity based on interaction data. 
Step 3: User's interest points should be a well-described sentence rather than a single word or phrase.
Provide concise explanation for each step, and rank the user's interests by importance, finally output the user's Final interests within a limited length. 
Output format: 'Explanation:\n Final interests:\n 
- Interest 1:\n - Interest 2:\n"""

OUTPUT_HAS_INTEREST_PROMPT_TEMPLATE = """Based on the user's current interaction data and previous interest summary, infer the user's preferences.\n
Step 1: Infer current preferences, focusing on item types the user may like.\n
Step 2: Compare with previous interests to check consistency or changes.\n
Step 3: Express interests as clear sentences, not single words.\n
Provide concise explanation for each step, and rank the user's interests by importance, finally output the user's Final interests within a limited length.
Output format: 'Explanation:\n Final interests:\n 
- Interest 1:\n - Interest 2:\n"""


STAGE_ACTION_TEMPLATE = ["""Using the current stage's interaction data, if prior user interests conflict or don't appear, remove relevant parts. In case of complete conflict with current interests, output current ones. Output format: Explanation:[Reason for action]\n Final interests:\n- Interest 1:\n - Interest 2:\n\n'""",
                         """Based on the interaction data at the current stage, if certain aspects of the previous user interests need some fine-tuning or modification at this stage, please indicate how to make the modifications. And then output the modified Final interests. If there is no need for modification, output the previous interests as they are. Note that you are only allowed to make fine-tunings and not add or delete anything new. Output format: 'Explanation:\n Final interests:\n- Interest 1:\n - Interest 2:\n'""",
                         """If new user interests are discovered in the interaction data at the current stage when compared with the previous interests, please add them to the previous ones. If there are no new interests, output the previous interests as they are. Please rank the user's interests by importance. Output format: 'Explanation:\n Final interests:\n- Interest 1:\n - Interest 2:\n'""",
                         """If the current stage's interests are consistent with the previous stage's and there is no significant change detected, then simply output the previous user interests without any alteration. Output format: 'Explanation:[Reason for action]\n. Final interests:\n- Interest 1:\n - Interest 2:\n"""]

DELETED_INTEREST_TEMPLATE = """
The summary of the user's interests in the previous stage is as follows: {interests}. 
The user's interactive behaviors in the current stage are as follows: {behaviors}. 
Please identify the abnormal behaviors of the user in the current stage.
Output format: # abnormal behavior ID: [such as No.n, No.n+x]
"""

ROUTER_TEMPLATE = """Judge based on user's current stage behavior and previous interests. 
First, identify main interests in the current stage by carefully considering items' kinds and characteristics. Note that same-category items with different features may reflect different user interests.
Then decide whether to delete, modify, add to previous interests or keep them unchanged. 
Note that only one such operation (deletion, modification, or addition) should be carried out per stage. 
Output format: Explanation:[Reason for action].\n Instruction:[only number, 1 for deletion, 2 for modification, 3 for addition, 4 for no change]."""

BASE_INPUT = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": "final_prompt"}
    ]

# Define function to create history message template
def create_history_message(row, index):
    if args.dataset == "beauty":
        message_parts = []
        if not pd.isna(row['item_title']):
            message_parts.append(f"No.{row['interaction_order']} title: {row['item_title']} categories: {row['item_categories']}") 
        else:
            if not pd.isna(row['item_description']):
                message_parts.append(f"description: {row['item_description']}")
            else:
                if not pd.isna(row['item_categories']):
                    message_parts.append(f"categories: {row['item_categories']}")
        return " ".join(message_parts) + "\n"
    elif args.dataset == "fashion":
        message_parts = []
        if row['item_title'] != "":
            message_parts.append(f"No.{row['interaction_order']} Title: {row['item_title']}") 
        else:
            message_parts.append(f"{row['interaction_order']}.Title: ")
        return " ".join(message_parts) + "\n"
    elif args.dataset == "yelp":
        return f"No.{row['interaction_order']} Title: {row['item_title']}. \n"
    elif args.dataset == "steam":
        return f"No.{row['interaction_order']} Title: {row['item_title']}. genres: {row['item_genres']}. game specs: {row['item_specs']}.\n"

def remove_last_two_interactions(group):
    # Keep the first n-2 interactions for each user (remove the last two interaction records)
    if len(group) <3:
        return group
    return group[:-2]


def assign_stage_and_batch_dynamic(df, stage_size):
    # TODO: Index is changed
    """
    Dynamically assign batch_id based on user's total number of stages. Each batch contains users with the same number of stages.
    """
    # Step 1: Add stage_id
    df["stage_id"] = df.groupby("user_id").cumcount() // stage_size

    # Step 2: Calculate total number of stages for each user
    user_stage_counts = df.groupby("user_id")["stage_id"].max() + 1
    user_stage_counts = user_stage_counts.reset_index().rename(
        columns={"stage_id": "num_stages"}
    )

    # Step 3: Assign batch_id based on num_stages
    user_stage_counts["batch_id"] = (
        user_stage_counts["num_stages"].rank(method="dense").astype(int) - 1  # dense method assigns the same rank to equal values without skipping ranks, from smallest to largest. For example, if two users both have num_stages=3, they will both be assigned rank 1, and the next different value (e.g., 4) will be assigned rank 2.
    )

    # Step 4: Merge batch_id back to original DataFrame
    original_index = df.index
    df = df.merge(
        user_stage_counts[["user_id", "batch_id"]], on="user_id", how="left"
    )
    df.index = original_index
    return df


def read_n_process_df(df_path, args):
    df = pd.read_csv(df_path, sep="\t")
    if args.is_filter_user_id:
        df = df[df["user_id"] == args.filter_user_id]
    
    df_sorted = df.sort_values(by=['user_id', 'interaction_order'])
    df_filtered = df_sorted.groupby('user_id', group_keys=False).apply(remove_last_two_interactions)
    return df_filtered


def extract_instruct_from_llm(tokenizer, init_seq_lens, generated_ids):
    filtered_outputs = [text[init_seq_lens:] for text in generated_ids]
    previous_translated_text = tokenizer.batch_decode(
        filtered_outputs, skip_special_tokens=True
    )
    valid_results, origin_results = [], []
    for result in previous_translated_text:
        origin_results.append(result)
        rindex = result.rfind("Instruction")
        if rindex == -1:  # Not found
            rindex = 0
        # Use regex to find numbers in text
        match = re.search(r'\d', result[rindex:])
        if match:
            num_str = match.group()
            if num_str in ['1', '2', '3', '4']:
                valid_results.append(int(num_str))
            else:
                print("not matched")
                print(result)
                valid_results.append(4)
        else:
            valid_results.append(4)
    return valid_results, origin_results


def get_first_non_value_indices(tensor, value=151643):
    # Find whether each element is not equal to the specified value
    non_value_mask = tensor != value
    # Add an extra True for each row to avoid the case where entire row is False
    padding = torch.ones((tensor.shape[0], 1), dtype=torch.bool, device=tensor.device)
    non_value_mask_with_padding = torch.cat((non_value_mask, padding), dim=1)
    # Convert boolean tensor to numeric type
    non_value_mask_with_padding = non_value_mask_with_padding.to(torch.float32)
    # Get index of first non-specified value
    indices = torch.argmax(non_value_mask_with_padding, dim=1)
    # If entire row is the specified value, index will be the last position, set it to -1
    all_value_mask = ~torch.any(non_value_mask, dim=1)
    indices[all_value_mask] = -1
    return indices


def extract_outputs_from_llm_for_delete(tokenizer, init_seq_lens, generated_ids):
    filtered_outputs = [text[init_seq_lens:] for text in generated_ids]
    previous_translated_text = tokenizer.batch_decode(
        filtered_outputs, skip_special_tokens=True
    )

    return previous_translated_text

def extract_outputs_from_llm(tokenizer, init_seq_lens, generated_ids):
    filtered_outputs = [text[init_seq_lens:] for text in generated_ids]
    previous_translated_text = tokenizer.batch_decode(
        filtered_outputs, skip_special_tokens=True
    )
    pure_previous_translated_text = []
    for text in previous_translated_text:
        rindex = text.rfind("Final interests")
        if rindex == -1:  # Not found
            rindex = 0
        pure_previous_translated_text.append(text[rindex+16:])
    # real_seq = tokenizer(
    #     pure_previous_translated_text, return_tensors="pt", padding=True, truncation=True
    # ).to(f"cuda:{args.gpu_id}")
    # real_seq_lens = get_first_non_value_indices(real_seq["input_ids"]) 
    # real_seq: length of newly generated content for each sample in a batch

    return previous_translated_text, pure_previous_translated_text

def check_and_create_folders(base_dir):
    # Define list of folders to check
    required_folders = ['last', 'mean', 'text', "mean_rnn","text_rnn"]
    
    for folder in required_folders:
        folder_path = os.path.join(base_dir, folder)
        try:
            # Use exist_ok=True to avoid exception if folder already exists
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder '{folder_path}' exists or created successfully.")
        except OSError as e:
            print(f"Error creating folder: {e}")


def generate(infos, tokenizer, model):
    texts = tokenizer.apply_chat_template(
        infos, tokenize=False, add_generation_prompt=True
    )
    seq = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    init_seq_lens = seq["input_ids"].shape[1]
    outputs = model.generate(
        **seq, max_new_tokens=1024, return_dict_in_generate=True, output_hidden_states=True
    )  # cache_implementation="static"
    return init_seq_lens, outputs

def main(df_path, save_path):
    check_and_create_folders(save_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/lgr/LLM4MSR/Qwen", trust_remote_code=True
    )
    # tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            "/home/lgr/LLM4MSR/Qwen", torch_dtype=torch.float16
        ).to(device)
    )
    model.eval()

    start_time = time.time()

    df_filtered = read_n_process_df(df_path, args)
    df = assign_stage_and_batch_dynamic(df_filtered, args.span_len)

    total_batches = len(df[df["batch_id"] >= args.terminate].groupby("batch_id"))
    df_group = df.groupby("batch_id")
    # Group by batch_id, users in each batch_group are different, but users in the same batch have the same stage_num
    for batch_id, batch_group in tqdm(
        df_group,
        desc="Processing Batches",
        total=total_batches,
        leave=False,
    ):
        if batch_id < args.terminate:
            continue
        batch_group = batch_group.sort_values(by=["user_id", "interaction_order"])
        user_ids = batch_group["user_id"].unique()
        user_data = batch_group.groupby("user_id")  # Group stages for each user
        max_stage = batch_group["stage_id"].max()  # Get maximum number of stages in current batch
        temp_last_user_preference_dict = {}
        temp_mean_user_preference_dict = {}
        # Refined summary for users in current batch
        previous_translated_text_dict = {
            str(user_id): "" for user_id in user_ids
        }  
        # Final summary for users in current batch
        user_interest_dict = {str(user_id): {} for user_id in user_ids}
        # user_interest_dict = dict.fromkeys(map(str, user_ids), {})

        # Process stage by stage, from stage_id = 0,1,2,3,4,5,...,
        for stage_index in tqdm(range(max_stage + 1), desc="Processing Stages in Batch", leave=False):
            # Data for all users with stage_id = i in current batch
            stage_data = batch_group[
                batch_group["stage_id"] == stage_index
            ] 
            # Maximum number of users per inference, batch inference saves time
            for start_idx in tqdm(range(0, len(user_ids), args.max_sub_batch_size),desc="Processing inner Batches",leave=False):  # Because number of users in each batch differs, some batches may have many users, processing all at once may cause memory overflow
                sub_batch_user_ids = user_ids[
                    start_idx : start_idx + args.max_sub_batch_size
                ]
                sub_batch_data = stage_data[
                    stage_data["user_id"].isin(sub_batch_user_ids)
                ]

                # Generate Prompt based on sub_batch_data
                prompts = []
                behavior_lists = {}
                span_user_ids = []
                for user_id, user_data in sub_batch_data.groupby("user_id"):

                    # Generate prompt for user history records
                    user_history_series = [create_history_message(row, index+1) for index, (_,row) in enumerate(user_data.iterrows())]
                    user_history_prompt = " ".join(user_history_series)
                    # Generate prompt based on stage
                    if stage_index == 0:
                        final_prompt = (
                            ENVIRONMENT_PROMPT_TEMPLATE
                            + user_history_prompt
                            + OUTPUT_NO_INTEREST_PROMPT_TEMPLATE
                        )
                    else:
                        last_stage_interest = previous_translated_text_dict[str(user_id)]
                        final_prompt = (
                            ENVIRONMENT_PROMPT_TEMPLATE
                            + user_history_prompt
                            + USER_PREVIOUS_INTERESTS_TEMPLATE.format(
                                last_stage_interest
                            )
                        ) 
                    # message = [
                    #     {
                    #         "role": "system",
                    #         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                    #     },
                    #     {"role": "user", "content": final_prompt}
                    # ]
                    BASE_INPUT[1]["content"] = final_prompt
                    message = copy.deepcopy(BASE_INPUT)
                    prompts.append(message)
                    span_user_ids.append(str(user_id))
                    behavior_lists[str(user_id)] = user_history_prompt
                # LLM batch inference - Vector output
                with torch.no_grad():
                    if stage_index == 0:
                        init_seq_lens, outputs = generate(prompts, tokenizer, model)
                        previous_translated_text, pure_previous_translated_text = extract_outputs_from_llm(tokenizer, init_seq_lens, outputs.sequences)

                        for i, user_id in enumerate(span_user_ids):
                            if stage_index not in user_interest_dict[user_id]:
                                user_interest_dict[user_id][stage_index] = {}
                            # TODO: There might be an issue here
                            user_interest_dict[user_id][stage_index]['behavior'] = behavior_lists[user_id]
                            # user_interest_dict[user_id][stage_index]["q_"+"0"] = USER_PREVIOUS_INTERESTS_TEMPLATE.format(previous_translated_text_dict[user_id]) 
                            user_interest_dict[user_id][stage_index]["final"] = previous_translated_text[i]
                        extracted_interest_dict = dict(zip(span_user_ids, pure_previous_translated_text))
                        previous_translated_text_dict.update(extracted_interest_dict)
                    else:
                        # Router inference
                        temp_prompts = None
                        if args.router_decision:
                            temp_prompts = copy.deepcopy(prompts)
                            for i, prompt in enumerate(temp_prompts):
                                prompt[1]["content"] += ROUTER_TEMPLATE
                            init_seq_lens, outputs = generate(temp_prompts, tokenizer, model)
                            valid_instructs, origin_instructs = extract_instruct_from_llm(tokenizer, init_seq_lens, outputs.sequences)
                            # Execute specific instructions
                            temp_prompts = copy.deepcopy(prompts)
                            for i, instruct in enumerate(valid_instructs):
                                temp_prompts[i][1]["content"] += STAGE_ACTION_TEMPLATE[instruct-1]

                        else:
                            temp_prompts = copy.deepcopy(prompts)
                            for i, prompt in enumerate(temp_prompts):
                                prompt[1]["content"] += OUTPUT_HAS_INTEREST_PROMPT_TEMPLATE
                                
                        init_seq_lens, outputs = generate(temp_prompts, tokenizer, model)
                        previous_translated_text, pure_previous_translated_text = extract_outputs_from_llm(tokenizer, init_seq_lens, outputs.sequences)
                        
                        if args.deleted == True:
                            deleted_prompts = []
                            for i, instruct in enumerate(valid_instructs):
                                if (instruct-1) == 0:  # Delete
                                    temp_deleted_prompt = DELETED_INTEREST_TEMPLATE.format_map({"interests": previous_translated_text_dict[span_user_ids[i]], "behaviors": behavior_lists[span_user_ids[i]]})
                                    BASE_INPUT[1]["content"] = temp_deleted_prompt
                                    deleted_prompts.append(copy.deepcopy(BASE_INPUT))
                            if len(deleted_prompts) > 0:
                                deleted_init_seq_lens, deleted_outputs = generate(deleted_prompts, tokenizer, model)
                                deleted_item_ids = extract_outputs_from_llm_for_delete(tokenizer, deleted_init_seq_lens, deleted_outputs.sequences)
                        
                        temp_num = 0   
                        for j, user_id in enumerate(span_user_ids):
                            if stage_index not in user_interest_dict[user_id]:
                                user_interest_dict[user_id][stage_index] = {}
                            if args.deleted and (valid_instructs[j] - 1) == 0:
                                user_interest_dict[user_id][stage_index]['deleted'] = deleted_item_ids[temp_num]
                                temp_num += 1
                            user_interest_dict[user_id][stage_index]['behavior'] = behavior_lists[user_id]
                            user_interest_dict[user_id][stage_index]["final"] = previous_translated_text[j]
                            if args.router_decision:
                                user_interest_dict[user_id][stage_index]['router'] = origin_instructs[j]
                                user_interest_dict[user_id][stage_index]["instruct"] = valid_instructs[j]

                        extracted_interest_dict = dict(zip(span_user_ids, pure_previous_translated_text))
                        previous_translated_text_dict.update(extracted_interest_dict)
                    # TODO: Does this only count generated tokens, excluding original ones?
                    generated_hidden_states = outputs.hidden_states  # len = 109 corresponds to length of generated tokens. Each element has len=29. Each element's shape is [16, 375, 3584], other elements' shape is [16, 1, 3584]
                    last_hidden_layer = [hidden[-1][:, -1, :] for hidden in generated_hidden_states]
                    mean_hidden_states = torch.stack(last_hidden_layer).mean(dim=0)
                    # TODO: Which is faster, saving directly to file or converting to CPU first?
                    mean_hidden_states = mean_hidden_states.detach().cpu().numpy()

                    # last_hidden_states = (
                    #     generated_hidden_states[-1][-1][:, -1, :]
                    #     .detach()
                    #     .cpu()
                    #     .numpy()  # vector_outputs.hidden_states[-1].shape = torch.Size([32, 390, 3584])
                    # )  # [max_sub_batch_size, 4096]
                    
                    del generated_hidden_states
                    # Update user_preference_dict
                    for key, mean_vector in zip(span_user_ids, mean_hidden_states):
                        # vector_dict = {key: last_vector}
                        mean_dict = {key: mean_vector}
                        # user_preference_dict.update(vector_dict)
                        # temp_last_user_preference_dict.update(vector_dict)
                        temp_mean_user_preference_dict.update(mean_dict)
                torch.cuda.empty_cache()

        if args.is_saved_text:
            with open(f"{save_path}/text_rnn/{batch_id}.json", 'w', encoding='utf-8') as f:
                json.dump(user_interest_dict, f, ensure_ascii=False, indent=4)
            # pickle.dump(user_interest_dict, open(f"{save_path}/interest/{batch_id}.pkl", "wb"))

        if args.is_saved_emb:
            # torch.save(
            #     temp_last_user_preference_dict,
            #     f"{save_path}/last/{batch_id}.pt",
            # )
            torch.save(
                temp_mean_user_preference_dict,
                f"{save_path}/mean_rnn/{batch_id}.pt",
            )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")


if __name__ == "__main__":
    # Use relative path, relative to project root directory
    df_path = f"./prompt/data/{args.dataset}/{args.dataset}.csv"
    save_path = f"./prompt/data/{args.dataset}"
    main(df_path=df_path, save_path=save_path)
