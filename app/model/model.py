import sys
import os
import time
from pathlib import Path

import torch


from auto_gptq import AutoGPTQForCausalLM
from langchain import PromptTemplate
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from pymilvus import connections
from transformers import AutoTokenizer, TextStreamer, pipeline

# Milvus connection
MILVUS_DB_NAME = "ge_1"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. But if even a little info is known, please do give an answer citing the source of the information.
""".strip()

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you absolutely cannot answer the question, you can say 'I don't know' or 'I don't understand'."


class ChatBot:
    def __init__(self):
        self.status = "Not Ready"
        self.status_log = []

    def connect_to_milvus(self):
        try:
            connections.connect(
                host=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DB_NAME
            )
        except Exception as excp:
            self.status = "Milvus_connection_failed"
            sys.exit(1)
        print("Milvus connected")
        self.status = "Connected_to_Milvus"
        self.status_log.append(self.status)

    def load_documents(self):
        loader = PyPDFDirectoryLoader(f"{Path(__file__).parent.parent}/pdfs")
        self.docs = loader.load()
        print("Documents loaded")
        self.status = "Documents_loaded"
        self.status_log.append(self.status)

    def load_embeddings(self):
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print("Embeddings loaded")
        self.status = "Embeddings_loaded"
        self.status_log.append(self.status)

    def load_text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=4)
        self.texts = text_splitter.split_documents(self.docs)
        print("Texts split")
        self.status = "Text_splitter_loaded"
        self.status_log.append(self.status)

    def load_vector_store(self):
        self.vector_store = Milvus.from_documents(
            self.texts,
            embedding=self.embeddings,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT,
                "db_name": MILVUS_DB_NAME,
            },
        )
        print("Vector store loaded")
        self.status = "Vector_store_loaded"
        self.status_log.append(self.status)

    def load_model(self):
        model_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        model_basename = "model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False,
            device=DEVICE,
            quantize_config=None,
        )
        print("Model loaded")
        self.status = "Model_loaded"
        self.status_log.append(self.status)

    def load_pipeline(self):
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True
        )
        self.text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
            streamer=self.streamer,
        )
        self.llm = HuggingFacePipeline(
            pipeline=self.text_pipeline, model_kwargs={"temperature": 0.1}
        )
        print("Pipeline loaded")
        self.status = "Pipeline_loaded"
        self.status_log.append(self.status)

    def _generate_prompt(self):
        def generate_prompt(
            prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
        ) -> str:
            return f"""
        [INST] <>
        {system_prompt}
        <>

        {prompt} [/INST]
        """.strip()

        template = generate_prompt(
            """
        {context}

        Question: {question}
        """,
            system_prompt=SYSTEM_PROMPT,
        )

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def load_qa_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self._generate_prompt()},
        )
        print("QA chain loaded")
        self.status = "QA_chain_loaded"
        self.status_log.append(self.status)

    def load_chatbot(self):
        self.connect_to_milvus()
        self.load_documents()
        self.load_embeddings()
        self.load_text_splitter()
        self.load_vector_store()
        self.load_model()
        self.load_pipeline()
        self.load_qa_chain()
        print("Chatbot loaded")
        self.status = "Ready_for_queries"
        self.status_log.append(self.status)

    def get_status(self):
        return self.status

    def generate_response(self, query):
        start = time.time()
        result = self.qa_chain(query, return_only_outputs=True)
        return {"response": result["answer"], "time_taken": time.time() - start}

    def model_shutdown(self):
        print("Model shutdown")
        sys.exit(1)

    def get_status_log(self):
        return self.status_log
