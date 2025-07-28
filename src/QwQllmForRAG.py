from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.huggingface import HuggingFaceLLM
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
import sys

# 本地运行大模型，使用llamaIndex实现RAG，并提供api接口的大模型服务
app = FastAPI()

class Query(BaseModel):
    query_txt: str

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# data为语料库，从指定目录data路径，读取文档，将数据加载到内存,#如果data的文档中有图片表格，需要调用OCR
documents = SimpleDirectoryReader("D:\\workspace\\ai\\rag\\data", required_exts=[".xlsx", ".txt", ".csv", ".json"]).load_data()

# 指定了一个预训练的sentence-transformer模型的路径
Settings.embed_model = resolve_embed_model("local:D:\\workspace\\ai\\rag\\model\\embedding\\gte-large-zh")

# 解析文档，并创建索引
index = VectorStoreIndex.from_documents(documents)
# 存储索引到默认地址
index.storage_context.persist()

#使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="D:\\workspace\\ai\\rag\\model\\Qwen2-0.5B-Instruct",
                tokenizer_name="D:\\workspace\\ai\\rag\\model\\Qwen2-0.5B-Instruct",
                device_map="auto",
                max_new_tokens=2048,
                context_window=30000,
                generate_kwargs={"temperature": 0.6, "top_k": 50, "top_p": 0.95},
                model_kwargs={"trust_remote_code":True},
                tokenizer_kwargs={"trust_remote_code":True})

# 设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

@app.post("/query")
async def query(query_txt : Query):
    prompt = query_txt.query_txt
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(prompt)
    return {"status": "success", "response": str(response)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)