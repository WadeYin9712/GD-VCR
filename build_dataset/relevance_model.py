import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import nltk
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
nltk.download("stopwords")
from nltk.corpus import stopwords
import random
import time
import json

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def text_preprocessing(x, objects):
    GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                            'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
    final_x = []
    for token in x:
        if isinstance(token, list):
            for idx in token:
                if objects[idx] == "person":
                    if idx >= len(GENDER_NEUTRAL_NAMES):
                        idx = len(GENDER_NEUTRAL_NAMES) - 1
                    final_x.append(GENDER_NEUTRAL_NAMES[idx])
                else:
                    final_x.append(objects[idx])
        else:
            final_x.append(token)
            
    return final_x

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True
            )
        
        input_ids.append(encoded_sent.get('input_ids')[:64])
        attention_masks.append(encoded_sent.get('attention_mask')[:64])

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Create the BertClassfier class
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
def initialize_model(train_dataloader, epochs=3):
    bert_classifier = BertClassifier(freeze_bert=False)
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=3e-5,
                      eps=1e-8
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, optimizer, scheduler, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        if evaluation == True:
            val_loss, val_accuracy = bert_predict(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
        
    model_path = "relevance_model_repro.th"
    torch.save(model, model_path)
    
    print("Training complete!")

def bert_predict(model, test_dataloader):
    model.eval()

    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def main():
    f = open("../X_VCR/train.jsonl", "r")
    
    final_txt = []
    final_txt_pos = []
    final_txt_neg = []
    q_list = []
    answer_cand = []
    y_train = []
    for line in f:
        data = json.loads(line)
        
        q = data["question"]
        ans_choices = data["answer_choices"]
        ans_id = data["answer_label"]
        ans_right = ans_choices[ans_id]
        objects = data["objects"]
        
        final_q = text_preprocessing(q, objects)
        final_ans = text_preprocessing(ans_right, objects)
        
        final_q_txt = ' '.join(final_q)
        final_ans_txt = ' '.join(final_ans)
        q_list.append(final_q_txt)
        answer_cand.append(final_ans_txt)
        final_txt_pos.append(final_q_txt + ' [SEP] ' + final_ans_txt)
    
    for i in range(len(final_txt_pos)):
        rand_idx = i
        while rand_idx == i:
            rand_idx = random.randint(0, len(answer_cand)-1)
        final_txt_neg.append(q_list[i] + ' [SEP] ' + answer_cand[rand_idx])
        final_txt.append(final_txt_pos[i])
        final_txt.append(final_txt_neg[i])
        
        y_train.append(1)
        y_train.append(0)
    
    # Concatenate train data and test data
    X_train = np.array(final_txt)
    y_train = np.array(y_train)
    
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    train_labels = torch.tensor(y_train)
    batch_size = 32
    
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=3)
    train(bert_classifier, train_dataloader, optimizer, scheduler, 
          epochs=3, evaluation=False)
    
if __name__ == "__main__":
    main()
