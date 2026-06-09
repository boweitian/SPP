import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForVision2Seq
from dataset import COCOVal2017Dataset, transform  # custom dataset
from tqdm import tqdm  # for progress bar
from PIL import Image

# 1. Device setup
cuda_num = 1
DEVICE = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"

# 2. Load model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b", load_in_8bit=True, device_map={"": cuda_num}  # enable 8-bit quantization
)

# 3. Load dataset
root_dir = "data/COCO/val2017"
dataset = COCOVal2017Dataset(root_dir=root_dir, transform=transform)

train_size = 64
test_size = len(dataset) - train_size

train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))  # first 64 images as training set
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))  # remaining images as test set

# 5. Create DataLoader
def collate_fn(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)  # keep PIL Image format

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 6. Process data and perform inference
def generate_captions(dataloader, split_name="train"):
    model.eval()
    results = []
    latent_features = []
    
    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            images = list(images)  # ensure images are List[PIL Image]
            
            # 1. Create prompt for each image
            prompts = [
                processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Pretend you're an honest person making statements about the world."}
                            ]
                        }
                    ],
                    add_generation_prompt=True
                ) for _ in images  # ensure each image gets its own message
            ]

            # 2. Process input
            inputs = processor(text=prompts, images=images, return_tensors="pt")  # batch_size images together
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # 3. Obtain latent representation via forward()
            outputs = model(**inputs, output_hidden_states=True)  # return hidden states
            latent = outputs.hidden_states[-1]  # get the last hidden state
            latent_features.append(latent.cpu().numpy())  # store to list
            
            # 3. Generate text
            generated_ids = model.generate(**inputs, max_new_tokens=50, temperature=2.0, do_sample=True)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # 4. Save results
            for img_id, caption in zip(image_ids, generated_texts):
                results.append({"image_id": img_id, "caption": caption})

    return results

# 7. Generate captions for training and test sets
train_results = generate_captions(train_loader, "train")

import json
with open("file/coco.json", "w") as f:
    json.dump(train_results, f, indent=4)
    
# test_results = generate_captions(test_loader, "test")
# with open("test_captions.json", "w") as f:
#     json.dump(test_results, f, indent=4)

print("Caption generation complete. Results saved.")
