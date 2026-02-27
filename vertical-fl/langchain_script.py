from dotenv import load_dotenv, find_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv(find_dotenv())
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import pandas as pd
from datasets import load_from_disk
import shap

from vertical_fl.task import (FEATURE_COLUMNS, 
    TARGET_COLUMN, DATASET_DIR, SEED, ServerModel, evaluate_head_model, TASK_TYPE, OUTPUT_SIZE, PARTITION_SIZES, DATASET_NAME, MODEL_FAMILY)



def main():
    full_english_labels = [
        "User-to-Root Privilege Escalation Attack",
        "Brute Force Authentication Attack",
        "Distributed Denial of Service Attack",
        "Denial of Service Attack",
        "Network Reconnaissance / Probing Attack",
        "Normal Benign Network Traffic",
        "Web Application Exploitation Attack",
        "Botnet-Based Malware Activity"
    ]

    subset_size = -1
    num_rounds = 40
    llm_text_gen = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.1,
        max_new_tokens=256,
    )
    llm_conversation = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="conversational",
        temperature=0.1,
        max_new_tokens=256,
    )

    chat = ChatHuggingFace(llm=llm_text_gen)

    prompt = ChatPromptTemplate(
        [
            ("system", "You are a network security expert with expertise in how different types of network threats can be identified and what mitigation steps are most critical for each."),
            ("human", "Explain plausible reasons why the intrusion detection system identified a {concept} attack in a couple lines, and give 3 recommended mitigation steps in a couple lines each.")
        ]
    )

    print(prompt)
    chain = prompt | chat
    #response = chain.invoke({"concept": "Distributed Denial of Service"})
    #print(response.content)

    model_name = f"{DATASET_NAME}_{MODEL_FAMILY}_vfl_{subset_size}sa_{num_rounds}eps"

    metadata = np.load(
        f"./server_model/{model_name}_metadata.npy",
        allow_pickle=True
    ).item()

    print(metadata)

    if TASK_TYPE == "binary":
        model = ServerModel(input_size=metadata["input_size"])
    else:  # multiclass
        model = ServerModel(input_size=metadata["input_size"], num_classes=metadata["num_classes"])

    state_dict = torch.load(f"./server_model/{model_name}_state.pt")
    model.load_state_dict(state_dict)

    emb_np = np.load(f"./server_model/testing_embeddings_{model_name}.npy", allow_pickle=True)
    X_emb = torch.from_numpy(emb_np).float()
    X_emb.requires_grad = True

    model.eval()

    # Background sample
    #g = torch.Generator()
    #g.manual_seed(SEED)
    #num_background = 100
    #indices = torch.randperm(X_emb.shape[0], generator=g)[:num_background]
    #background = X_emb[indices]
    #explainer = shap.DeepExplainer(model, background)

    userInput = input("Enter an index: ")

    try:
        userIndex = int(userInput)
        print(f"You entered a valid integer: {userIndex}")
    except ValueError:
        print("Input is not an integer, continuing...")
        userIndex = -1
    while (userInput != "n" and userIndex != -1):
        x_index = X_emb[userIndex:userIndex+1]

        with torch.no_grad():
            output = model(x_index)

        probs = torch.softmax(output, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        full_english_label = full_english_labels[pred_class_idx]

        response = chain.invoke({"concept": full_english_label})
        print(response.content)

        #shap_values = explainer.shap_values(x_index, check_additivity=False)
        userInput = input("Enter an index: ")

        try:
            userIndex = int(userInput)
            print(f"You entered a valid integer: {userIndex}")
        except ValueError:
            print("Input is not an integer, continuing...")
            userIndex = -1

if __name__ == "__main__":
    main()