import os
import torch
import transformers
import torch
from datasets import load_dataset
import os
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import fire
import torch.nn as nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator()

def tensor_loss(tensor1,tensor2):
    j = 0
    for i in range(tensor1.shape[1]):
        if torch.equal(tensor1[0][i],tensor2[0][i]):
            j += 1
    return j/tensor1.shape[1]

def grounder(tensor):
    max_values, _ = torch.max(tensor, dim=2, keepdim=True)
    binary_mask = torch.where(tensor == max_values, torch.tensor(1.0), torch.tensor(0.0))

    return binary_mask

def collate_fn(batch):
    max_seq_length = max(len(sample[0]) for sample in batch)
    padded_batch = [torch.nn.functional.pad(sample[0], (0, max_seq_length - len(sample[0]))) for sample in batch]
    return torch.stack(padded_batch), 0


config = {"tokenizer": {
        "max_length": 1024,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    }}

# device = accelerator.device

class DistillDataset(Dataset):

    def __init__(self, dataset_name, model,tokenizer):
        self.dataset = load_dataset(dataset_name,split = "train_sft")
        self.model = model
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input = self.tokenizer.apply_chat_template(
                    self.dataset[idx]['messages'],
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )['input_ids'].to("cuda")
        return input

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

def train(model_id :str,dataset_name :str,epochs :int):
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )
    # model = pipeline.model
    # model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    dataset = load_dataset(dataset_name,split="train_sft")

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_2 = nn.MSELoss()

    dataset = DistillDataset(dataset_name,model,tokenizer)
    data_loader = DataLoader(dataset,collate_fn=collate_fn,batch_size = 1)

    total_loss = 0
    
    model, optimizer, data_loader = accelerator.prepare(
    model, optimizer, data_loader
    )
    model.train()
    for epoch in range(epochs):
        for i,input in enumerate(data_loader):
            # tensor_tuple = ()

            # for j in range(len(os.listdir("logits_messages_1"))):
            #     tensor = torch.load(f"logits_messages_{i+1}/tensor{j+1}.pt")
            #     tensor_tuple = tensor_tuple + (tensor,)

            # stacked_tensor = torch.stack(tensor_tuple,dim = 0)
            # teacher_logits = torch.stack((stacked_tensor,),dim = 0)
            # tensor_tuple = ()

            # for j in range(len(os.listdir("ground_truth_1"))):
            #     tensor = torch.load(f"ground_truth_{i+1}/tensor{j+1}.pt")
            #     tensor_tuple = tensor_tuple + (tensor,)
            print(input)
            print("Hello")
            ground_truth_input = F.one_hot(input[0])
            
            # ground_truth = torch.stack(tensor_tuple,dim = 0)
            # ground_truth = torch.stack((ground_truth,),dim = 0)
            teacher_logits = torch.load(f"Tensors1/output{i}.pt")
            # teacher_logits = torch.stack((logits_tensor,),dim = 0)
            tensor_array = []
            print("problem")
            outputs = model(input[0])
            print("Solve")
            original_loss = outputs.loss
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
            student_logits = concatenated_tensor
            # student_logits = torch.stack((concatenated_tensor,),dim = 0)
            # student_logits = outputs.logits
            # student_logits = (torch.sort(outputs.logits[0][j]))[0]
            # student_logits = student_logits[:20]
            # student_logits_ground = student_logits[:1]
            print(student_logits.shape,teacher_logits.shape)
            student_logits,teacher_logits = pad_logits(student_logits,teacher_logits)
            # student_logits_ground,ground_truth = pad_logits(student_logits_ground,ground_truth)
            print(student_logits.shape,teacher_logits.shape)
            student_logits_scaled = student_logits / config["distillation"]["temperature"]
            teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]
            student_logits_scaled = torch.stack((student_logits_scaled,),dim = 0)
            teacher_logits_scaled = torch.stack((teacher_logits_scaled,),dim = 0)
            # ground_truth = ground_truth / config["distillation"]["temperature"]
            # student_logits_ground = student_logits_ground/ config["distillation"]["temperature"]
            print(student_logits_scaled.shape,teacher_logits_scaled.shape)
            student_logits_GC,ground_truth_input = pad_logits(torch.stack((student_logits,),dim=0),ground_truth_input)
            print("Start")
            print(student_logits_GC.shape,ground_truth_input.shape)
            print(grounder(student_logits_GC),ground_truth_input)
            print("stop")
            loss_kd = F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction='batchmean'
            ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]
            loss_2 = loss = tensor_loss(grounder(student_logits_GC),ground_truth_input)
            # loss_ = loss_2(student_logits_scaled,ground_truth)
            # loss = loss_kd + loss_
            # total_loss += loss
            loss_real = config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"])
            # optimizer.zero_grad()
            loss = loss_2 + loss_real
            accelerator.backward(loss)
            # loss_real.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            del student_logits,teacher_logits,student_logits_scaled,teacher_logits_scaled,concatenated_tensor, tensor_array,outputs
            torch.cuda.empty_cache()
    
            print(f"Loss = {loss} for epoch = {epoch}")

if __name__ == "__main__":
    fire.Fire()
               