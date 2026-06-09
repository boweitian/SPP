import json
import random
import numpy as np
from transformers import PreTrainedTokenizerFast

def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizerFast, seed: int = 0):
    """
    Processes COCO dataset to create a dataset where each sample contains:
    - Truncated statements
    - Honest and untruthful variations
    - Corresponding image_id

    Args:
    - data_path (str): Path to the JSON containing the data.
    - tokenizer (PreTrainedTokenizerFast): Tokenizer to tokenize statements.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Dictionary containing all processed data under a single key.
    """

    random.seed(seed)

    with open(data_path, "r") as f:
        data = json.load(f)

    template_str = "USER: <image> Pretend you're {type} person making statements about the world. ASSISTANT:"

    dataset = [] 

    for item in data:
        image_id = item["image_id"] 
        statement = item["caption"].split("Assistant:")[-1].strip() 
        tokens = tokenizer.tokenize(statement)

        for idx in range(1, len(tokens) - 5): 
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_text = f"{template_str.format(type='an honest')} {truncated_statement}"
            untruthful_text = f"{template_str.format(type='an untruthful')} {truncated_statement}"

            dataset.append({
                "image_id": image_id,
                "honest": honest_text,
                "untruthful": untruthful_text
            })

    print(f"Total processed samples: {len(dataset)}")

    return {"data": dataset}

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased") 
dataset = honesty_function_dataset("file/coco.json", tokenizer)

with open("file/coco_honesty_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
