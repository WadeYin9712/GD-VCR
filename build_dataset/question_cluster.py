import nltk
import json
from sentence_transformers import SentenceTransformer
import math
import matplotlib.pyplot as plt
import random
import torch
import time
import numpy as np
import os
 
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)

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
 
def k_means(dataset, k, iteration):
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(dataset[i])
        
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
        
    while iteration > 0:
        start = time.time()
        C = []
        for i in range(k):
            C.append([])
        for labelIndex, item in enumerate(dataset):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if dist < minDist:
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(dataset[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
        end = time.time()
        print(end-start, iteration)
    return C, labels

device = torch.device("cpu")

model = SentenceTransformer('stsb-roberta-base')

def make_list(filename):
    f = open(filename, "r")
    questions = []
    questions_orig = []
    
    for line in f:
        data = json.loads(line)
        objects = data["objects"]
        final_question = text_preprocessing(data["question"], objects)
        pos_tag_result = nltk.pos_tag(final_question)
        questions.append(' '.join([final_question[0]] + [y for (x,y) in pos_tag_result]))
        questions_orig.append(final_question)
        # print(questions[-1])
        
    return questions, questions_orig


question_list, questions_orig_list = make_list("val.jsonl")

if not os.path.exists("cluster.npy"):
    question_embeddings = model.encode(question_list, convert_to_tensor=True).to(device)
    question_embeddings = question_embeddings.cpu().numpy().tolist()
    
    C, labels = k_means(question_embeddings, 10, 200)
    
    np.save("cluster.npy", np.array(C))
    np.save("question_labels.npy", np.array(labels))
else:
    C = np.load("cluster.npy", allow_pickle=True)
    labels = np.load("question_labels.npy", allow_pickle=True)
    clusters = []
    
    for i, label in enumerate(labels):
        if label == 0:
            print(' '.join(questions_orig_list[i]), label)








