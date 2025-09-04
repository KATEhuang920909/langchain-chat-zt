import asyncio
import json
import os
from typing import AsyncIterable, List, Optional, Any

import pandas as pd
from fastapi import Body, File, Form, UploadFile
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from sse_starlette.sse import EventSourceResponse

from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, BM25_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from server.chat.utils import History
from server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.knowledge_base.kb_service.bm25_service import BM25Service
from server.knowledge_base.utils import KnowledgeFile
from server.utils import (wrap_done, get_ChatOpenAI,
                          BaseResponse, get_prompt_template, get_temp_dir, run_in_thread_pool)


def _parse_files_in_thread(
        files: List[UploadFile],
        dir: str,
        zh_title_enhance: bool,
        chunk_size: int,
        chunk_overlap: int,
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """

    def parse_file(file: UploadFile) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # 读取上传文件的内容

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            return True, filename, f"成功上传文件 {filename}", docs
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result


def _parse_pkg_in_thread(
        files: List[Any],
        dir: str,
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """

    def parse_file(file):
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # 读取上传文件的内容

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)

            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            kb_file.loader_kwargs = {"extract_path": dir}
            parse_info, pkg_info = kb_file.file2docs()
            return True, filename, parse_info, pkg_info
        except Exception as e:
            msg = f"{filename} 文件解压失败，报错信息为: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result


def _parse_log_in_thread(
        files: List[Any],
        dir: str,
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """

    def parse_file(file):
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # 读取上传文件的内容

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)

            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            parse_info, pkg_info = kb_file.file2dataframe()
            return True, filename, parse_info, pkg_info
        except Exception as e:
            msg = f"{filename} 文件解压失败，报错信息为: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result

def upload_temp_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        prev_id: str = Form(None, description="前知识库ID"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents = []
    path, id = get_temp_dir(prev_id)
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                           dir=path,
                                                           zh_title_enhance=zh_title_enhance,
                                                           chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})

    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    return BaseResponse(data={"id": id, "failed_files": failed_files})


def upload_temp_docs_v2(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        prev_id: str = Form(None, description="前知识库ID"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents_info = []
    documents = []
    path, id = get_temp_dir(prev_id)
    # True, filename, f"成功上传文件 {filename}", docs
    # print("upload_temp_docs_v2_files", files)
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                           dir=path,
                                                           zh_title_enhance=zh_title_enhance,
                                                           chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap):
        # if success:
        #     documents += docs
        # else:
        #     failed_files.append({file: msg})

        if success:
            documents_info.append({'文件名': file})
            documents.append(docs)
            # documents.append(docs)
        else:
            failed_files.append({file: msg})

    # with memo_faiss_pool.load_vector_store(id).acquire() as vs:
    #     vs.add_documents(documents)
    if documents_info:
        new_df = pd.DataFrame(documents_info)
        # st.success(f"✅ 成功添加 {len(documents_info)} 个文件")
    return BaseResponse(
        data={"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })


def upload_temp_pkgfile(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        prev_id: str = Form(None, description="前知识库ID")) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents_info = []
    documents = []
    path, id = get_temp_dir(prev_id)
    # True, filename, f"成功上传文件 {filename}", docs
    # print("upload_temp_docs_v2_files", files)
    for success, file, msg, docs in _parse_pkg_in_thread(files=files, dir=path):
        # if success:
        #     documents += docs
        # else:
        #     failed_files.append({file: msg})

        if success:
            documents_info.append({'文件名': file})
            documents.append(docs)
            # documents.append(docs)
        else:
            failed_files.append({file: msg})

    # with memo_faiss_pool.load_vector_store(id).acquire() as vs:
    #     vs.add_documents(documents)
    if documents_info:
        new_df = pd.DataFrame(documents_info)
        # st.success(f"✅ 成功添加 {len(documents_info)} 个文件")
    return BaseResponse(
        data={"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })


def upload_temp_logfile(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        prev_id: str = Form(None, description="前知识库ID")) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents_info = []
    documents = []
    path, id = get_temp_dir(prev_id)
    # True, filename, f"成功上传文件 {filename}", docs
    # print("upload_temp_docs_v2_files", files)
    for success, file, msg, docs in _parse_log_in_thread(files=files, dir=path):
        # if success:
        #     documents += docs
        # else:
        #     failed_files.append({file: msg})

        if success:
            documents_info.append({'文件名': file})
            documents.append(docs)
            # documents.append(docs)
        else:
            failed_files.append({file: msg})

    # with memo_faiss_pool.load_vector_store(id).acquire() as vs:
    #     vs.add_documents(documents)
    if documents_info:
        new_df = pd.DataFrame(documents_info)
        # st.success(f"✅ 成功添加 {len(documents_info)} 个文件")
    return BaseResponse(
        data={"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })


async def file_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                    knowledge_id: str = Body(..., description="临时知识库ID"),
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                    score_threshold: float = Body(SCORE_THRESHOLD,
                                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                  ge=0, le=2),
                    history: List[History] = Body([],
                                                  description="历史对话",
                                                  examples=[[
                                                      {"role": "user",
                                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                      {"role": "assistant",
                                                       "content": "虎头虎脑"}]]
                                                  ),
                    stream: bool = Body(False, description="流式输出"),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                    prompt_name: str = Body("default",
                                            description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                    ):
    if knowledge_id not in memo_faiss_pool.keys():
        return BaseResponse(code=404, msg=f"未找到临时知识库 {knowledge_id}，请先上传文件")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        embed_func = EmbeddingsFunAdapter()
        embeddings = await embed_func.aembed_query(query)
        with memo_faiss_pool.acquire(knowledge_id) as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [x[0] for x in docs]

        context = "\n".join([doc.page_content for doc in docs])
        if len(docs) == 0:  ## 如果没有找到相关文档，使用Empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator())


async def file_chat_v2(query: str = Body(..., description="用户输入", examples=["你好"]),
                       knowledge_id: str = Body(..., description="临时知识库ID"),
                       documents: list = Body(..., description="临时知识库切片文档"),
                       top_k: int = Body(BM25_SEARCH_TOP_K, description="字符串向量数"),
                       history: List[History] = Body([],
                                                     description="历史对话",
                                                     examples=[[
                                                         {"role": "user",
                                                          "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                         {"role": "assistant",
                                                          "content": "虎头虎脑"}]]
                                                     ),
                       stream: bool = Body(False, description="流式输出"),
                       model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                       temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                       max_tokens: Optional[int] = Body(None,
                                                        description="限制LLM生成Token数量，默认None代表模型最大值"),
                       prompt_name: str = Body("default",
                                               description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                       ):
    # if knowledge_id not in memo_faiss_pool.keys():
    #
    #     return BaseResponse(code=404, msg=f"未找到临时知识库 {knowledge_id}，请先上传文件")
    if knowledge_id:
        history = [History.from_data(h) for h in history]
        bm25_model = BM25Service()
        print("bm25_model", bm25_model)

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        # embed_func = EmbeddingsFunAdapter()
        # embeddings = await embed_func.aembed_query(query)
        # with memo_faiss_pool.acquire(knowledge_id) as vs:
        #     docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
        #     docs = [x[0] for x in docs]
        # bm25检索

        _ = [[bm25_model.add_document(d["page_content"]) for d in doc] for doc in documents]
        doc_list = []
        __ = [[doc_list.append(d) for d in doc] for doc in documents]
        bm25_model.calculate_idf()
        docs = bm25_model.search(query)
        bm25_result = [doc_list[idx_score[0]] for idx_score in docs[:top_k] if idx_score[1]]
        bm25_score = [idx_score for idx_score in docs[:top_k]]
        print("bm25_score", bm25_score)
        print("bm25_result", bm25_result)
        context = "||".join([k["page_content"] for k in bm25_result])

        if len(bm25_result) == 0:  ## 如果没有找到相关文档，使用Empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        print({"context@@@@@@@": context, "question": query})
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(bm25_result):
            filename = doc["metadata"].get("source")
            text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc["page_content"]}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator())
