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

from vertical_fl.task import (FEATURE_COLUMNS, 
    TARGET_COLUMN, DATASET_DIR, SEED, ServerModel, evaluate_head_model, TASK_TYPE, OUTPUT_SIZE, PARTITION_SIZES, DATASET_NAME, MODEL_FAMILY)



def main():
    subset_size = -1
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
            ("human", "Explain plausible reasons why the intrusion detection system identified a {concept} attack in a couple lines and give 3 recommended mitigation steps in a couple lines each.")
        ]
    )

    print(prompt)
    chain = prompt | chat
    response = chain.invoke({"concept": "Distributed Denial of Service"})
    print(response.content)

    metadata = np.load(
        "./server_model/my_model_metadata.npy",
        allow_pickle=True
    ).item()

    print(metadata)

    if TASK_TYPE == "binary":
        model = ServerModel(input_size=metadata["input_size"])
    else:  # multiclass
        model = ServerModel(input_size=metadata["input_size"], num_classes=metadata["num_classes"])

    dataset = load_from_disk(DATASET_DIR)

    # Currently the entire dataset is in a pseudo "train" split
    dataset = dataset["train"].train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    if 0 < subset_size < len(dataset["train"]):
        test_dataset = dataset["test"].select(range(subset_size))
    else:
        test_dataset = dataset["test"]

if __name__ == "__main__":
    main()