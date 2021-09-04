import torch
import numpy as np
from transformers import BertModel
from sentence_transformers import SentenceTransformer, util
from relevance_model import text_preprocessing, preprocessing_for_bert, bert_predict, BertClassifier
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import time
import os
import math
import random

# Generate answer candidates (refer to Alg. in Appendix)
def generate_answer_choices(i, relevance_results, similarity, similarity_answers, 
                            choices, 
                            itera, thres=0.9, thres1=0.2):
    candidates = []
    min_score = 100000000
    bucket_size = int(len(relevance_results[0])/3)
    
    for j in range(bucket_size):
        if relevance_results[i][j+itera*bucket_size] > thres:
            candidates.append(j+itera*bucket_size)
    
    score = 0
    min_idx = -1
    if itera == 0:
        for candidate in candidates:
            score = similarity[i][candidate]
            if similarity[i][candidate] < thres1:
                score = 10
            if score < min_score:
                min_score = score
                min_idx = candidate
    else:
        for candidate in candidates:
            score = similarity[i][candidate]
            if similarity[i][candidate] < thres1:
                score = 10
            for choice in choices:
                if similarity_answers[choice][candidate] < thres1:
                    score += 10
                else:
                    score += similarity_answers[choice][candidate]
            if score < min_score:
                min_score = score
                min_idx = candidate
        
    return min_idx

# Obtain text data
def make_list(filename):
    f = open(filename, "r")
    answers = []
    questions = []
    questions_with_idx = []
    answer_choices = []
    answer_choices_idx = []
    
    for line in f:
        data = json.loads(line)
        questions_with_idx.append(data["question"])
        ans_choices = data["answer_choices"]
        ans_id = data["answer_label"]
        ans_right = ans_choices[ans_id]
        objects = data["objects"]
        
        final_question = text_preprocessing(data["question"], objects)
        final_ans = text_preprocessing(ans_right, objects)
        answers.append(' '.join(final_ans))
        questions.append(' '.join(final_question))
        answer_choices.append(ans_right)
        answer_choices_idx.append(ans_id)
        
    return answers, questions, answer_choices, questions_with_idx, answer_choices_idx

# Replace the object indices in answer candidates with the ones existing in the right answer 
def limit_range(choices, objects_size):
    new_choices = []
    for choice in choices:
        new_choice = []
        for token in choice:
            if isinstance(token, list):
                idx_list = []
                for idx in token:
                    if idx >= objects_size:
                        idx_list.append(objects_size-1)
                    else:
                        idx_list.append(idx)
                new_choice.append(idx_list)
            else:
                new_choice.append(token)
        new_choices.append(new_choice)
    
    return new_choices

# Integrate the original dataset with only right answers
def make_final_list(filename, choices, choices_idx):
    f = open(filename, "r")
    
    with open("../X_VCR/val.jsonl", 'w') as f1:
        i = 0
        for line in f:
            data = json.loads(line)

            # Erroreous DP
            if "west_17.jpg" in data["img_fn"]:
                i += 1
                continue
            data["answer_choices"] = limit_range(choices[i], len(data["objects"]))
            data["answer_label"] = choices_idx[i]
            data["rationale_choices"] = limit_range(choices[i], len(data["objects"]))
            data["rationale_label"] = choices_idx[i]
            data_json = json.dumps(data)
            f1.write(data_json+'\n')
            i += 1

def remove_zero(arr):
    arr_new = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 0:
                arr_new.append(arr[i][:j])
                break
    return arr_new

# Convert answer choices into VCR form
def convert_to_VCR(x, idx):
    pronouns = ["he", "she", "it", "they", "his", "her", "their", "him", "them", "its"]
    arr = x.split(" ")
    arr_new = []
    idx_list = []
    
    for token in arr:
        if token in GENDER_NEUTRAL_NAMES:
            idx_list.append(name2idx[token])
        else:
            if idx_list != []:
                arr_new.append(idx_list)
                idx_list = []
            arr_new.append(token)
            
    if idx_list != []:
        arr_new.append(idx_list)
        
    question_idx = []               
    for token in questions_with_idx_MC[idx]:
        if isinstance(token, list):
            question_idx.append(token)
            
    orig_choices = answer_choices_MC[idx]
    
    i = 0
    j = 0
    final = []
    while i < len(arr_new) and j < len(orig_choices):
        if isinstance(arr_new[i], list):
            while not isinstance(orig_choices[j], list):
                j += 1
                if j == len(orig_choices):
                    j -= 1
                    break
            
            if isinstance(orig_choices[j], list):
                final.append(orig_choices[j])
                j += 1
            else:
                if len(question_idx) != 0:
                    final.append(question_idx[0])
                    question_idx = question_idx[1:]
                else:
                    final.append(arr_new[i])
        elif arr_new[i].lower() in pronouns:
            while not isinstance(orig_choices[j], list):
                j += 1
                if j == len(orig_choices):
                    j -= 1
                    break
            
            if isinstance(orig_choices[j], list):
                final.append(orig_choices[j])
                if arr_new[i].lower() == "his" or arr_new[i].lower() == "her" or arr_new[i].lower() == "their" or arr_new[i].lower() == "its":
                    final.append('\'')
                    final.append('s')
                j += 1
            else:
                final.append(arr_new[i])
        else:
            final.append(arr_new[i])
            
        i += 1
                    
    return final

model = SentenceTransformer('stsb-roberta-base')

answers_MC, questions_MC, answer_choices_MC, questions_with_idx_MC, answer_choices_idx_MC = make_list("MC-VCR_test.jsonl")
answers, _, _, _, _ = make_list("../X_VCR/orig_val.jsonl")

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

name2idx = dict()
for i, name in enumerate(GENDER_NEUTRAL_NAMES):
    name2idx[name] = i

if not os.path.exists("relevance.npy"):
    relevance_model = torch.load("relevance_model.th")
    relevance_model.eval()
    
    questions_MC_tokenized, questions_attention_mask_MC = preprocessing_for_bert(questions_MC)
    answers_MC_tokenized, attention_mask_MC = preprocessing_for_bert(answers_MC)
    answers_tokenized, attention_mask = preprocessing_for_bert(answers)
    
    questions_MC_tokenized = remove_zero(questions_MC_tokenized.numpy().tolist())
    answers_MC_tokenized = remove_zero(answers_MC_tokenized.numpy().tolist())
    answers_tokenized = remove_zero(answers_tokenized.numpy().tolist())
    questions_attention_mask_MC = remove_zero(questions_attention_mask_MC.numpy().tolist())
    attention_mask_MC = remove_zero(attention_mask_MC.numpy().tolist())
    attention_mask = remove_zero(attention_mask.numpy().tolist())
    
    relevance_results = np.zeros((len(answers_MC_tokenized), len(answers_tokenized)))
    
    # Compute the relevance matrix w.r.t. all the right choices of GD-VCR and VCR dev
    for i, sample_MC in enumerate(questions_MC_tokenized):
        start = time.time()
        batch_size = 64
        val_inputs = [sample_MC + sample[1:] for sample in answers_tokenized]
        val_masks = [questions_attention_mask_MC[i] + sample[1:] for sample in attention_mask]
        val_inputs_1 = [val_input[:64] if len(val_input) > 64 else val_input + [0] * (64-len(val_input)) for val_input in val_inputs]
        val_masks_1 = [val_mask[:64] if len(val_mask) > 64 else val_mask + [0] * (64-len(val_mask)) for val_mask in val_masks]
        val_inputs = torch.tensor(val_inputs_1)
        val_masks = torch.tensor(val_masks_1)
        val_labels = torch.tensor(np.array([0] * len(answers)))
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        results = bert_predict(relevance_model, val_dataloader)
        relevance_results[i] = np.array([result[1] for result in results])
        end = time.time()
        print(i, len(results), len(results[0]), end - start)
        
    np.save("relevance.npy", relevance_results)
else:
    embeddings1 = model.encode(answers_MC, convert_to_tensor=True)
    embeddings2 = model.encode(answers, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    device = torch.device("cpu")
    relevance_results = np.load("relevance.npy")
    
    embeddings_answers = model.encode(answers, convert_to_tensor=True)
    similarity_answers = util.pytorch_cos_sim(embeddings_answers, embeddings_answers)
    
    choices = []
        
    for i in range(len(relevance_results)):
        choices.append([])
        for itera in range(3):
            choices[i].append(generate_answer_choices(i, relevance_results, similarity, similarity_answers, choices[i], itera))
    
    final_choices = []
    final_choices_idx = []
    
    for i in range(len(choices)):
        final_choices.append([])
        for j, choice in enumerate(choices[i]):
            if j == answer_choices_idx_MC[i]:
                final_choices[i].append(answer_choices_MC[i])
            final_choices[i].append(convert_to_VCR(answers[choice], i))
        if answer_choices_idx_MC[i] == 3:
            final_choices[i].append(answer_choices_MC[i])
        print(final_choices[i])
        
        final_choices_idx.append(random.randint(0,3))
        temp = final_choices[-1][answer_choices_idx_MC[i]]
        final_choices[-1][answer_choices_idx_MC[i]] = final_choices[-1][final_choices_idx[-1]]
        final_choices[-1][final_choices_idx[-1]] = temp
        print(final_choices_idx[-1])
        print(final_choices[i])
        
        assert len(final_choices[i]) == 4
        print("----")
        
    make_final_list("MC-VCR_test.jsonl", final_choices, final_choices_idx)
