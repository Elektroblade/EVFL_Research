from dotenv import load_dotenv, find_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatHuggingFace
load_dotenv(find_dotenv())
from langchain_core.messages import SystemMessage, HumanMessage


def main():
    model_id = "meta-llama/Llama-3.1-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
    )

    chat = ChatHuggingFace(pipeline=pipe)

    response = chat("Explains LLMs in one sentence.")
    print(response)

    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain LLMs in one sentence.")
    ]
    chat(messages)


if __name__ == "__main__":
    main()