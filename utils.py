from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import FAISS
from typing import Optional, List
import os


class ProxyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt


def init_knowledge_vector_store(filepath: str, embeddings):
    md_splitter = MarkdownTextSplitter(chunk_size=256, chunk_overlap=0)
    docs = []
    if not os.path.exists(filepath):
        print("路径不存在")
        return None
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
            print(f"{file} 已成功加载")
        except:
            print(f"{file} 未能成功加载")
            return None
    elif os.path.isdir(filepath):
        for file in os.listdir(filepath):
            fullfilepath = os.path.join(filepath, file)
            try:
                loader = UnstructuredFileLoader(fullfilepath, mode="elements")
                docs += loader.load()
                print(f"{file} 已成功加载")
            except:
                print(f"{file} 未能成功加载")

#    texts = md_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def init_chain_proxy(llm_proxy: LLM, vector_store, top_k=5):
#     prompt_template = """你是一个专业的图书馆参考咨询员，请你基于以下已知信息，完整和专业的回答用户的问题。
#     如果无法从中得到答案，请说"不好意思，作为一个AI咨询员我暂时无法回答，我可以为您转接人工服务"，不允许在答案中添加编造成分，答案请使用中文。
#
# 已知内容:
# {context}
#
# 参考以上内容请回答如下问题:
# 请从以上几段内容中挑选一段作为答案参考，不要参考多个答案，回答如下问题:
# {question}"""

    prompt_template = """{context}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"]
    )
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm_proxy,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    return knowledge_chain

def init_chain_proxy_1(llm_proxy: LLM, vector_store, top_k=5):
#     prompt_template = """你是一个专业的图书馆参考咨询员，请你基于以下已知信息，完整和专业的回答用户的问题。
#     如果无法从中得到答案，请说"不好意思，作为一个AI咨询员我暂时无法回答，我可以为您转接人工服务"，不允许在答案中添加编造成分，答案请使用中文。
#
# 已知内容:
# {context}
#
# 参考以上内容请回答如下问题:
# 请从以上几段内容中挑选一段作为答案参考，不要参考多个答案，回答如下问题:
# {question}"""

    prompt_template = """你是一个咨询员，请你基于以下已知信息，准确和专业的回答用户的问题。
        如果无法从以下内容中得到答案，请说"不好意思，作为一个AI咨询员我暂时无法回答，我可以为您转接人工服务"，不允许在答案中添加编造成分，答案请使用中文。

    已知内容:
    {context}

    下面会给你一个问题请你回答，请从以上几段内容中挑选一段最贴切问题的内容作为答案参考，尽量不要参考多个答案:
    {question}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm_proxy,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    return knowledge_chain
