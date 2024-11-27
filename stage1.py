import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import os
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.quantization import quantize_dynamic

#model_id = "meta-llama/Meta-Llama-3-8B"

from accelerate import Accelerator

accelerator = Accelerator()

def collate_fn(batch):
    max_seq_length = max(len(sample[0]) for sample in batch)
    padded_batch = [torch.nn.functional.pad(sample[0], (0, max_seq_length - len(sample[0]))) for sample in batch]
    return torch.stack(padded_batch), 0

def main(dataset_name :str, student_model_id :str):
    model_id = student_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",load_in_8bit=True)

    class DistillDataset(Dataset):

        def __init__(self, dataset_name):
            self.dataset = load_dataset(dataset_name,split = "train_sft")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            input = tokenizer.apply_chat_template(
                        self.dataset[idx]['messages'],
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )['input_ids'].to("cuda")
            return input

    dataset = DistillDataset(dataset_name)
    data_loader = DataLoader(dataset,collate_fn=collate_fn,batch_size = 1)
    model, data_loader = accelerator.prepare(
    model, data_loader
    )
    for i,input in enumerate(data_loader):
        tensor_array = []
        with torch.inference_mode():
            outputs = model(input[0])
        # os.mkdir(f"Tensors/logits_messages_{i+1}")
        # os.mkdir(f"Tensors/ground_truth_{i+1}")
        for j in range(outputs.logits[0].shape[0]):
            logits = (torch.sort(outputs.logits[0][j]))[0]
            logits = torch.flip(logits,[0])
            new_logits = logits[:20]
            tensor_array.append(new_logits)
            # ground_truth = logits[:1]
            # torch.save(new_logits,f"Tensors/logits_messages_{i+1}/tensor{j+1}.pt")
            # torch.save(ground_truth,f"Tensors/ground_truth_{i+1}/tensor{j+1}.pt")
            del logits, new_logits
            torch.cuda.empty_cache()
        concatenated_tensor = torch.stack(tensor_array)
        print(concatenated_tensor)
        print("Done")
        torch.save(concatenated_tensor,f"Tensors1/output{i}.pt")
        del concatenated_tensor
        torch.cuda.empty_cache()