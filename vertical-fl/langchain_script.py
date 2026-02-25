from dotenv import load_dotenv, find_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatHuggingFace
load_dotenv(find_dotenv())
from langchain_core.messages import SystemMessage, HumanMessage


def main():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        task="text-generation",
        temperature=0.1,
        max_new_tokens=256,
    )

    chat = ChatHuggingFace(llm=llm)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain LLMs in one sentence.")
    ]

    response = chat.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()