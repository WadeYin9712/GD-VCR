"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil
import json
from copy import deepcopy

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from allennlp.nn.util import device_mapping
from vis import grounding_vis


from visualbert.utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, restore_checkpoint_flexible, load_state_dict_flexible, compute_score_with_logits
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

from visualbert.dataloaders.vcr import VCR, VCRLoader
try:
    from visualbert.dataloaders.coco_dataset import COCODataset
except:
    print("Import COCO dataset failed.")
try:   
    from visualbert.dataloaders.nlvr_dataset import NLVRDataset
except:
    print("Import NLVR2 dataset failed.")
try:
    from visualbert.dataloaders.vqa_dataset import VQADataset
except:
    print("Import VQA dataset failed.")
try:
    from visualbert.dataloaders.flickr_dataset import Flickr30kFeatureDataset
except:
    print("Import Flickr30K dataset failed.")

from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

from allennlp.models import Model
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.models import model
from attrdict import AttrDict

def check_prob(val_probs, INDEX_OF_CHECKED_SAMPLE):
    ps = np.exp(val_probs[INDEX_OF_CHECKED_SAMPLE])
    ps /= np.sum(ps)

    return ps

# If you want to play grounding analysis, feel free to use this function!
def grounding_analysis(args, input_batch, output_dict, question_orig, answer_orig, obj_added_index, \
                       file_name_list, annot_id_list, b):
    if args.orig_or_new == "new":
        bert_input_ids = input_batch["bert_input_ids"].detach().cpu().numpy()
        labels = input_batch["label"].detach().cpu().numpy()
        objects = input_batch["objects"].detach().cpu().numpy()
        attention_weights = output_dict["attention_weights"][-1].detach().cpu().numpy()
                    
        question_orig_cur = question_orig[b*args.eval_batch_size: b*args.eval_batch_size+len(bert_input_ids)]
        answer_orig_cur = answer_orig[b*args.eval_batch_size: b*args.eval_batch_size+len(bert_input_ids)]
        obj_added_index_cur = obj_added_index[b*args.eval_batch_size: b*args.eval_batch_size+len(bert_input_ids)]
        file_name_list_cur = file_name_list[b*args.eval_batch_size: b*args.eval_batch_size+len(bert_input_ids)]
        annot_id_list_cur = annot_id_list[b*args.eval_batch_size: b*args.eval_batch_size+len(bert_input_ids)]
                    
        if args.addition_annotation_analysis:
            for i in range(len(bert_input_ids)):
                label = labels[i]
                dets2use = dets2uses[i]
                file_name = file_name_list_cur[i]
                annot_id = annot_id_list_cur[i]
                right_ans_input_ids = bert_input_ids[i][label]
                attention_weights_i = attention_weights[i*4+label]
                                
                texts = tokenizer.convert_ids_to_tokens(right_ans_input_ids)
                texts, people_names = recover(texts, question_orig_cur[i], answer_orig_cur[i])
                j = 0
                obj_list = []
                for obj in objects[i]:
                    if obj == 0:
                        obj_list.append("[BG]")
                    elif obj == -1:
                        obj_list.append("[I_PAD]")
                    else:
                        obj_list.append("["+obj_added_index_cur[i][int(dets2use[j])]+"]\n(image)")
                        j += 1
                                
                texts += obj_list
                indices = []
                for j, token in enumerate(texts):
                    if token == "[CLS]" or token == "[SEP]" or token == "[PAD]" or token == "[I_PAD]" or token == ".":
                        indices.append(j)
                                
                texts = np.delete(texts, indices, axis=0)
                                
                for j in range(len(attention_weights_i)):
                    attention_temp = np.delete(attention_weights_i[j], indices, axis=0)
                    final_attention = np.delete(attention_temp, indices, axis=1)
                                
                    assert len(texts) == len(final_attention)
                                    
                    pos_seg = file_name.find('/')
                    file_name = file_name[pos_seg+1:]
                    grounding_vis(final_attention, texts, file_name.replace(".", "_"+annot_id+"_head_"+str(j)+"_result."), args.region, args.single_or_multiple)

def recover(texts, question_orig, answer_orig):
    all_orig = question_orig + answer_orig
    classes_orig = []
    people_names = []
    
    GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
    GENDER_NEUTRAL_NAMES_new = [name.lower() for name in GENDER_NEUTRAL_NAMES]
    
    for token in all_orig:
        if '[' in token and ']' in token:
            classes_orig.append(token)
            
    for i, token in enumerate(texts):
        if token in GENDER_NEUTRAL_NAMES_new:
            while "person" not in classes_orig[0]:
                if len(classes_orig) == 1:
                    classes_orig = []
                    break
                classes_orig = classes_orig[1:]
            
            if classes_orig:
                texts[i] = classes_orig[0]
                people_names.append(classes_orig[0])
                if len(classes_orig) >= 2:
                    classes_orig = classes_orig[1:]
    
    return texts, people_names

def add_index(obj_orig_list):
    added_index_all = []
    
    for obj_orig in obj_orig_list:
        added_index = []
        freq = dict()
        for obj in obj_orig:
            if obj not in freq.keys():
                freq[obj] = 1
            else:
                freq[obj] += 1
                
            added_index.append(obj+str(freq[obj]))
        added_index_all.append(added_index)
    
    return added_index_all

parser = argparse.ArgumentParser(description='train')

parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

parser.add_argument(
    '-config',
    dest='config',
    help='config location',
    type=str,
)

parser.add_argument(
    '-region',
    dest='region',
    default='any',
    help='region',
    type=str,
)

parser.add_argument(
    '-single_or_multiple',
    dest='single_or_multiple',
    default='single',
    help='single_or_multiple',
    type=str,
)

parser.add_argument(
    '-orig_or_new',
    dest='orig_or_new',
    default='new',
    help='orig_or_new',
    type=str,
)

parser.add_argument(
    '-addition_annotation_analysis',
    dest='addition_annotation_analysis',
    action='store_true',
)

parser.add_argument(
    '-grounding',
    dest='grounding',
    action='store_true',
)

parser.add_argument(
    '-scene',
    dest='scene',
    default='none',
    help='scene',
    type=str,
)

parser.add_argument(
    '-not_use_all_dets',
    dest='not_use_all_dets',
    action='store_false'
)

args = parser.parse_args()

args = ModelWrapper.read_and_insert_args(args, args.config)
#####################################################

if os.path.exists(args.folder):
    create_flag = 0
else:
    create_flag = 1
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)

import sys
run_log_counter = 0

while(os.path.exists(args.folder + '/run_{}.log'.format(run_log_counter))):
    run_log_counter += 1

file_log = open(args.folder + '/run_{}.log'.format(run_log_counter),'w')  # File where you need to keep the logs
file_log.write("")
class Unbuffered:
    def __init__(self, stream):
       self.stream = stream
    def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       file_log.write(data)    # Write the data of stdout here to a text file as well
    def flush(self):
        pass

sys.stdout = Unbuffered(sys.stdout)

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if args.get("fp16", False):
        _to_fp16(td)

    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            if td[k] is not None:
                td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(non_blocking=True)
    return td
def _to_fp16(td):
    for k in td:
        if isinstance(td[k], torch.FloatTensor):
            td[k] = td[k].to(dtype=torch.float16)

num_workers = args.get("num_workers", 2)
val_workers = args.get("val_workers", 0)

TEST_DATA_READING = False
if TEST_DATA_READING:
    num_workers = 0

print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

def get_dataset_loader(args, dataset_name):
    if dataset_name == "vcr":
        train, orig_val, val, val_addition, test = VCR.splits(
                                  mode='rationale' if args.rationale else 'answer',
                                  region_keywords = args.region,
                                  scene = args.scene,
                                  single_or_multiple = args.single_or_multiple,
                                  only_use_relevant_dets = args.not_use_all_dets,
                                  do_lower_case = args.do_lower_case,
                                  bert_model_name = args.bert_model_name,
                                  max_seq_length = args.max_seq_length,
                                  pretraining = args.pretraining,
                                  pretraining_include_qa_and_qar = args.pretraining_include_qa_and_qar,
                                  complete_shuffle = args.get("complete_shuffle", False),
                                  use_alignment = args.get('use_alignment', False),
                                  add_all_features = args.add_all_features,
                                  answer_labels_path = args.get("answer_labels_path", None),
                                  vcr_annots_dir = args.vcr_annots_dir,
                                  vcr_image_dir = args.vcr_image_dir
                                  )
    elif dataset_name == "coco":
        train, val, test = COCODataset.splits(args)
    elif dataset_name == "nlvr":
        train, val, test = NLVRDataset.splits(args)
    elif dataset_name == "vqa":
        train, val, test = VQADataset.splits(args)
    elif dataset_name == "flickr":
        train, val, test = Flickr30kFeatureDataset.splits(args)
    else:
        assert(0)

    loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
    train_loader_params = deepcopy(loader_params)
    loader_params_val = {'batch_size': args.eval_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
    val_loader_params = deepcopy(loader_params_val)
    val_loader_params["num_workers"] = val_workers
    test_loader_params = deepcopy(loader_params_val)
    test_loader_params["num_workers"] = val_workers
    
    train_loader = VCRLoader.from_dataset(train, **train_loader_params)
    val_loader = VCRLoader.from_dataset(val, **val_loader_params)
    test_loader = VCRLoader.from_dataset(test, **test_loader_params)
    
    if dataset_name == "vcr":
        orig_val_loader_params = deepcopy(loader_params_val)
        orig_val_loader_params["num_workers"] = val_workers
        val_addition_loader_params = deepcopy(loader_params_val)
        val_addition_loader_params["num_workers"] = val_workers
        orig_val_loader = VCRLoader.from_dataset(orig_val, **orig_val_loader_params)
        val_addition_loader = VCRLoader.from_dataset(val_addition, **val_addition_loader_params)
    
    train_set_size = len(train)
    
    print("orig_val size", len(orig_val))
    print("val size", len(val))
    print("val-addition size", len(val_addition))
    
    if dataset_name == "vcr":
        return train_loader, orig_val_loader, val_loader, val_addition_loader, test_loader, train_set_size
    else:
        return train_loader, val_loader, test_loader, train_set_size

print(args)

if args.dataset == "vcr":
    train_loader, orig_val_loader, val_loader, val_addition_loader, test_loader, train_set_size = get_dataset_loader(args, args.dataset)
else:
    train_loader, val_loader, test_loader, train_set_size = get_dataset_loader(args, args.dataset)


ARGS_RESET_EVERY = args.get("print_every", 100)


train_model = ModelWrapper(args, train_set_size)

# Loading from pre-trained model
if args.restore_bin:
    train_model.restore_checkpoint_pretrained(args.restore_bin)

# Loading from previous checkpoint
'''if create_flag == 0:
    start_epoch, val_metric_per_epoch = train_model.restore_checkpoint(serialization_dir=args.folder, epoch_to_load = args.get("epoch_to_load", None))
    if val_metric_per_epoch is None:
        val_metric_per_epoch = []
else:
    create_flag = 1
    start_epoch, val_metric_per_epoch = 0, []'''
    
start_epoch, val_metric_per_epoch = 0, []

shutil.copy2(args.config, args.folder) # Always copy the config

if args.get("freeze_detector", True):
    train_model.freeze_detector()

param_shapes = print_para(train_model.model)

print(args)

print("########### Starting from {}".format(start_epoch))

num_batches = 0
    
stop_epoch = args.num_train_epochs

save_every = args.get("save_every", None)

with open('../dataloaders/cocoontology.json', 'r') as f1:
    coco = json.load(f1)
coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]

tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, do_lower_case=args.do_lower_case)

for epoch_num in range(start_epoch, stop_epoch):
    train_results = []
    norms = []
    train_model.model.train()
    if not args.get("skip_training", False):
        for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
            del batch["dets2use"]
            batch = _to_gpu(batch)
            
            output_dict = train_model.step(batch)

            num_batches += 1

            train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                            'crl': output_dict.get("cnn_regularization_loss", 0.0),
                                            'next_sentence_loss': output_dict["next_sentence_loss"].mean().item() if "next_sentence_loss" in output_dict else 0.0,
                                            'masked_lm_loss': output_dict["masked_lm_loss"].mean().item() if "masked_lm_loss" in output_dict else 0.0,
                                            'accuracy': (train_model.model.module).get_metrics(
                                                reset=(b % ARGS_RESET_EVERY) == 0)[
                                                'accuracy'],
                                            'sec_per_batch': time_per_batch,
                                            'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                            }))
            if b % ARGS_RESET_EVERY == 0 and b > 0:
                print("e{:2d}b{:5d}/{:5d}. \nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                    epoch_num, b, len(train_loader),
                    pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
                ), flush=True)

            if save_every is not None and b % save_every == 0 and b != 0:
                train_model.save_checkpoint_step(args.folder, b, epoch_num)

        print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))

    try:
        ### This is the eval part
        val_probs = []
        val_labels = []
        val_size = 0.0
        val_loss_sum = 0.0

        val_acc = 0.0
        val_acc_upper = 0.0
        val_instance_counter = 0.0

        val_next_sentence_loss_sum = 0.0

        train_model.eval()

        val_counter = 0
        
        keywords_list = []
        regions_list = []
        
        if not args.skip_training:
            val_loader = orig_val_loader
            val_dataset = orig_val_loader.dataset
        else:
            if args.orig_or_new == "new":
                annot_fn = "val.jsonl"
                with open(os.path.join(args.vcr_annots_dir, annot_fn), 'r') as f:
                    all_items = [json.loads(s) for s in f]
                keywords_list = [it["keywords"] for it in all_items]
                regions_list = [it["region"] for it in all_items]
                
                if args.addition_annotation_analysis:
                    annot_fn = "val_addition_single.jsonl"
                    val_loader = val_addition_loader
                    val_dataset = val_addition_loader.dataset
                        
                if args.grounding:
                    question_orig = []
                    answer_orig = []
                    obj_orig_list = []
                    file_name_list = []
                    annot_id_list = []
                    items_temp = []
                    if args.region != "any":
                        for item in all_items:
                            if args.region in item["region"]:
                                items_temp.append(item)
                    else:
                        for item in all_items:
                            items_temp.append(item)
                    
                    if args.addition_annotation_analysis and args.single_or_multiple == "multiple":
                        image_fn_list = [item["img_fn"] for item in items_temp]
                        with open(os.path.join(args.vcr_annots_dir, 'val.jsonl'), 'r') as f:
                            temp_val = [json.loads(s) for s in f]
                            
                        temp = []
                        for item in temp_val:
                            if item["img_fn"] in image_fn_list:
                                temp.append(item)
                        
                        items_temp = temp
                    
                    if args.scene != "none":
                        for item in items_temp:
                            if args.scene in item["keywords"]:
                                question_orig.append(item["question_orig"])
                                answer_orig.append(item["answer_orig"])
                                obj_orig_list.append(item["objects"])
                                file_name_list.append(item["img_fn"])
                                annot_id_list.append(item["annot_id"])
                    else:
                        question_orig = [it["question_orig"] for it in items_temp]
                        answer_orig = [it["answer_orig"] for it in items_temp]
                        obj_orig_list = [it["objects"] for it in items_temp]
                        file_name_list = [it["img_fn"] for it in items_temp]
                        annot_id_list = [it["annot_id"] for it in items_temp]
                        
                    obj_added_index = add_index(obj_orig_list)
            else:
                val_loader = orig_val_loader
                val_dataset = orig_val_loader.dataset
            
        # for vqa, nlvr, flickr
        do_test = args.get("do_test", False) ## This one is for vqa
        if do_test:
            val_loader = test_loader
            val_dataset = val_loader.dataset
        vcr_save_result = args.get("vcr_save_result", False) # This one is for vcr

        for b, (time_per_batch, batch) in enumerate(time_batch(val_loader if args.no_tqdm else tqdm(val_loader), reset_every=ARGS_RESET_EVERY)):
            with torch.no_grad():
                input_batch = batch
                dets2uses = input_batch["dets2use"].detach().cpu().numpy()
                del input_batch["dets2use"]
                batch = _to_gpu(batch)
                output_dict = train_model.step(batch, eval_mode = True)
                
                if args.grounding:
                    grounding_analysis(args, input_batch, output_dict, question_orig, answer_orig, obj_added_index, file_name_list, annot_id_list, b)
                        
                if not args.pretraining:
                    if args.model.training_head_type == "vqa":
                        val_probs.append(output_dict['logits'].detach().cpu())
                        if not do_test:
                            val_labels.append(batch['label'].detach().cpu())
                    elif args.model.training_head_type == "flickr":
                        # This is because of multi-GPU
                        val_acc += (output_dict["accuracy"] * output_dict["entity_num"].float()).sum(-1).item()
                        val_acc_upper += (output_dict["upperbound_accuracy"] * output_dict["entity_num"].float()).sum(-1).item()
                        val_instance_counter += output_dict["entity_num"].sum(-1).item()

                    elif args.model.training_head_type == "multichoice":
                        val_probs.append(output_dict['logits'].detach().cpu().numpy())
                        if not do_test:
                            val_labels.append(batch['label'].detach().cpu().numpy())
                    elif args.model.training_head_type == "nlvr":
                        val_probs.append(output_dict['logits'].detach().cpu().numpy())
                        val_labels.append(batch['label'].detach().cpu().numpy())

                else:
                    val_labels.append(batch['label'].detach().cpu().numpy())

                if not do_test:
                    val_loss_sum += output_dict['loss'].mean().item() * batch['label'].size(0)
                    val_counter += batch['label'].size(0)

                    if "next_sentence_loss" in output_dict:
                        val_next_sentence_loss_sum += output_dict['next_sentence_loss'].mean().item() * batch['label'].size(0)

        if not args.pretraining:
            if args.model.training_head_type == "vqa":
                if do_test:
                    val_probs = np.concatenate(val_probs, 0)
                    val_probs = torch.Tensor(val_probs)
                    val_probs = val_probs.squeeze(1)
                    val_dataset.generate_test_file(val_probs, os.path.join(args.folder, "result.json"))
                    print("Finished testing")
                    assert(0)
                else:
                    val_labels = np.concatenate(val_labels, 0)
                    val_probs = np.concatenate(val_probs, 0)
                    val_probs = torch.Tensor(val_probs)
                    val_labels = torch.Tensor(val_labels)
                    val_probs = val_probs.squeeze(1)
                    acc = torch.sum(compute_score_with_logits(val_probs, val_labels)) / val_labels.size(0)
                    acc = acc.squeeze(-1).item()
            elif args.model.training_head_type == "flickr":
                acc = val_acc / val_instance_counter
                val_acc_upper = val_acc_upper / val_instance_counter
                print("Upper bound: {:.5f}".format(val_acc_upper))
            elif args.model.training_head_type == "multichoice": #VCR
                if not do_test:
                    val_labels = np.concatenate(val_labels, 0)
                val_probs = np.concatenate(val_probs, 0)
                
                # Stats for Figure 3 and Table 4
                if args.skip_training and args.region == "any" and args.scene == "none" and not args.addition_annotation_analysis and args.orig_or_new == "new":
                    category_total = dict()
                    category_total["east-asia"], category_total["south-asia"], category_total["west"], category_total["africa"] = dict(), dict(), dict(), dict()
                    for k, pred in enumerate(val_probs.argmax(1)):
                        # print(val_labels[k])
                        if int(pred) == val_labels[k]:
                            for keyword in keywords_list[k]:
                                if keyword not in category_total[regions_list[k]].keys():
                                    category_total[regions_list[k]][keyword] = dict()
                                    category_total[regions_list[k]][keyword]["corr"] = 1
                                    category_total[regions_list[k]][keyword]["tot"] = 1
                                else:
                                    category_total[regions_list[k]][keyword]["corr"] += 1
                                    category_total[regions_list[k]][keyword]["tot"] += 1
                        else:
                            for keyword in keywords_list[k]:
                                if keyword not in category_total[regions_list[k]].keys():
                                    category_total[regions_list[k]][keyword] = dict()
                                    category_total[regions_list[k]][keyword]["corr"] = 0
                                    category_total[regions_list[k]][keyword]["tot"] = 1
                                else:
                                    category_total[regions_list[k]][keyword]["tot"] += 1
                
                    for keyword in category_total["west"].keys():
                        if keyword in category_total["east-asia"].keys() and category_total["west"][keyword]["tot"] >= 8 and category_total["east-asia"][keyword]["tot"] >= 8:
                            print("west", keyword, category_total["west"][keyword])
                            print("east-asia", keyword, category_total["east-asia"][keyword])
                            print("----")
                        
                if vcr_save_result:
                    if do_test:
                        file_name = "test"
                    else:
                        file_name = "val"

                    save_file_name = os.path.join(args.folder, file_name + "_qa.np")
                    if args.rationale:
                        save_file_name = os.path.join(args.folder, file_name + "_qar.np")
                    if do_test:
                        np.save(save_file_name, val_probs)
                    else:
                        np.savez(save_file_name+'z', val_probs=val_probs, val_labels=val_labels)
                    
                    print("Saved result to {}".format(save_file_name))
                    assert(0)

                acc = float(np.mean(val_labels == val_probs.argmax(1)))
            elif args.model.training_head_type == "nlvr":
                val_labels = np.concatenate(val_labels, 0)
                val_probs = np.concatenate(val_probs, 0)
                if args.get("report", False):
                    val_probs = val_probs.argmax(1)
                    assert(val_probs.shape[0]) == len(val_dataset)
                    result = []
                    for index, i in enumerate(val_dataset.items):
                        label = "True" if val_probs[index] == 1 else "False"
                        result.append(i["identifier"] + "," + label)
                    with open(os.path.join(args.folder, "results.csv"), "w") as f:
                        f.write("\n".join(result))
                    assert(0)
                acc = float(np.mean(val_labels == val_probs.argmax(1)))
            if not do_test:
                val_loss_avg = val_loss_sum / val_counter
                print("Val epoch {} has acc {:.5f} and loss {:.5f}".format(epoch_num, acc, val_loss_avg), flush=True)
            else:
                print("Val epoch {} has acc {:.5f}".format(epoch_num, acc), flush=True)
                assert(0)
            val_metric_per_epoch.append(acc)
        else:
            val_loss_avg = val_loss_sum / val_counter
            val_next_sentence_loss_avg = val_next_sentence_loss_sum / val_counter
            print("Val epoch {} has loss {:.5f}, next sentence loss {:.5f}".format(epoch_num, val_loss_avg, val_next_sentence_loss_avg), flush=True)
            val_metric_per_epoch.append(-val_loss_avg)
        
        if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - args.patience):
            print("Stopping at epoch {:2d}".format(epoch_num))
            break
        ############### Save model
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, val_metric_per_epoch, is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))
    except KeyboardInterrupt:
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, None, is_best=False)
        print("Something Went Wrong with Evaluation. Stopped.")
        assert(0)
    except:
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, None, is_best=False)
        print("Something Went Wrong with Evaluation. Ignored.")
        if args.get("skip_training", False):
            assert(0)
