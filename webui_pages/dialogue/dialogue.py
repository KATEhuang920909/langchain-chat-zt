import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime
import os
import re
import time
from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
                     DEFAULT_KNOWLEDGE_BASE, BM25_SEARCH_TOP_K)
from server.knowledge_base.utils import LOADER_DICT, PKG_DICT, LOG_DICT
import uuid
from typing import List, Dict
import pandas as pd
from PIL import Image

chat_box = ChatBox(
    greetings=['欢迎使用能力中台安全测评工具，我们竭诚为您服务!\n\n'],
    assistant_avatar=os.path.join(
        "img",
        "回复者.png"
    ),
    user_avatar=os.path.join(
        "img",
        "提问者.png"
    )
)
# 向页面注入自定义 CSS
hide_drag_drop_style = """
    <style>
        /*  targeting the drag and drop area of st.file_uploader */
        div[data-testid="stFileUploaderDropzone"] {
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            padding: 10px;
            text-align: center;
            /* 尝试隐藏整个拖拽区域 */
            display: none !important;
        }
        /* 你可能还需要调整外部容器，确保“浏览文件”按钮仍然可见 */
        section[data-testid="stFileUploader"] > div:first-child {
            /* 调整样式，例如只让按钮显示 */
            border: none;
            padding: 0;
        }
        /* 确保“Browse files”按钮本身可见 */
        button[data-testid="baseButton-primary"] {
            /* 确保上传按钮的样式 */
            display: block !important;
        }
    </style>
"""
st.markdown(hide_drag_drop_style, unsafe_allow_html=True)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    '''
    return _api.upload_temp_docs_v2(files).get("data", {}).get("id")


@st.cache_data
def upload_temp_docs_v2(files, _api: ApiRequest) -> tuple:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    {"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })
    '''
    print("_api.upload_temp_docs_v2(files)", files)
    print("_api.upload_temp_docs_v2(files)", _api.upload_temp_docs_v2(files))

    data = _api.upload_temp_docs_v2(files).get("data", {})
    id, success_info, table_df, documents = data.get("id"), data.get("success_info"), data.get("table_df"), data.get(
        "documents")
    return id, success_info, table_df, documents


@st.cache_data
def upload_temp_pkgfile(files, _api: ApiRequest) -> tuple:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    {"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })
    '''
    print("_api.upload_temp_pkgfile(files)", files)
    print("_api.upload_temp_pkgfile(files)", _api.upload_temp_pkgfile(files))
    data = _api.upload_temp_pkgfile(files).get("data", {})
    id, success_info, table_df, documents = data.get("id"), data.get("success_info"), data.get("table_df"), data.get(
        "documents")
    return id, success_info, table_df, documents


@st.cache_data
def upload_temp_logfile(files, _api: ApiRequest) -> tuple:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    {"id": id,
              "failed_files": failed_files,
              "success_info": f"成功添加 {len(documents_info)} 个文件",
              "table_df": new_df if documents_info else "",
              "documents": documents
              })
    '''

    data = _api.upload_temp_logfile(files).get("data", {})
    print("_api.upload_temp_logfile(files)", data)
    id, success_info, table_df, documents = data.get("id"), data.get("success_info"), data.get("table_df"), data.get(
        "documents")
    return id, success_info, table_df, documents


def parse_command(text: str, modal: Modal) -> bool:
    '''
    检查用户是否输入了自定义命令，当前支持：
    /new {session_name}。如果未提供名称，默认为“会话X”
    /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
    /clear {session_name}。如果未提供名称，默认清除当前会话
    /help。查看命令帮助
    返回值：输入的是命令返回True，否则返回False
    '''
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            modal.open()
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"会话{i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"该会话名称 “{name}” 已存在")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("这是最后一个会话，无法删除")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"无效的会话名称：“{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
        return True
    return False


def dialogue_page(api: ApiRequest, is_lite: bool = False):
    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)

    default_model = api.get_default_llm_model()[0]

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用智能化安全运营交互式知识服务平台! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    # 弹出自定义命令帮助信息
    modal = Modal("自定义命令", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            st.write("\n\n".join(cmds))

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        conversation_name = st.selectbox("当前会话：", conv_names, index=index)
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)

        dialogue_modes = ["LLM 对话",
                          "知识库问答",
                          "文件对话",
                          "日志解析",
                          "材料匹配",
                          ]
        dialogue_mode = st.selectbox("请选择对话模式：",
                                     dialogue_modes,
                                     index=0,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )

        def on_llm_change():
            if llm_model:
                config = api.get_model_config(llm_model)
                if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                    st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        print("api.list_running_models", api.list_running_models())
        running_models = list(api.list_running_models())
        available_models = []
        config_models = api.list_config_models()
        if not is_lite:
            for k, v in config_models.get("local", {}).items():
                if (v.get("model_path_exists")
                        and k not in running_models):
                    available_models.append(k)
        for k, v in config_models.get("online", {}).items():
            if not v.get("provider") and k not in running_models and k in LLM_MODELS:
                available_models.append(k)
        llm_models = running_models + available_models
        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in llm_models:
            index = llm_models.index(cur_llm_model)
        else:
            index = 0
        llm_model = st.selectbox("选择LLM模型：",
                                 llm_models,
                                 index,
                                 format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model",
                                 )
        if (st.session_state.get("prev_llm_model") != llm_model
                and not is_lite
                and not llm_model in config_models.get("online", {})
                and not llm_model in config_models.get("langchain", {})
                and llm_model not in running_models):
            with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        index_prompt = {
            "LLM 对话": "llm_chat",
            "材料匹配": "llm_chat",
            "日志解析": "llm_chat",
            "知识库问答": "knowledge_base_chat",
            "文件对话": "knowledge_base_chat",
        }
        prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
        prompt_template_name = prompt_templates_kb_list[0]
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            text = f"已切换为 {prompt_template_name} 模板。"
            st.toast(text)

        prompt_template_select = st.selectbox(
            "请选择Prompt模板：",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
        )
        prompt_template_name = st.session_state.prompt_template_select
        temperature = st.slider("Temperature：", 0.0, 2.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

        if dialogue_mode == "知识库问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases()
                index = 0
                if DEFAULT_KNOWLEDGE_BASE in kb_list:
                    index = kb_list.index(DEFAULT_KNOWLEDGE_BASE)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    index=index,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, BM25_SEARCH_TOP_K)

                ## Bge 模型会超过1
                score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
        elif dialogue_mode == "文件对话":

            with st.expander("文件对话配置", True):
                files = st.file_uploader("上传知识文件：",
                                         [i for ls in LOADER_DICT.values() for i in ls if i != '.doc'],
                                         accept_multiple_files=True,
                                         )

                kb_top_k = st.number_input("匹配知识条数：", 1, 20, BM25_SEARCH_TOP_K)
                print("files", files)
                ## Bge 模型会超过1
                # score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                upload_button = st.button("开始上传", disabled=len(files) == 0)

                if upload_button:
                    st.session_state.setdefault("documents", [])
                    # st.session_state.setdefault("select_documents", [])
                    # st.session_state.setdefault("selected_rows", [])
                    #
                    st.session_state.setdefault("df_upload", pd.DataFrame([], columns=["文件名"]))

                    if len(files) != 0:
                        doc_id, success_info, table_df, documents = upload_temp_docs_v2(files, api)
                        # documents = [[k["page_content"] for k in doc] for doc in documents]
                        print("files", files)
                        print("success_info", success_info)
                        print("table_df", table_df)
                        print("documents", documents)
                        st.session_state["file_chat_id"] = doc_id
                        st.session_state["documents"] = documents
                        st.success(success_info)
                        st.session_state["df_upload"] = pd.DataFrame(table_df)
                    # selection = st.dataframe(st.session_state.df_upload,
                    #                          use_container_width=True,
                    #                          hide_index=True,
                    #                          on_select="rerun",
                    #                          selection_mode="multi-row",  # 允许多行选择
                    #                          key="my_dataframe",
                    #                          )
                    # selected_rows = selection["selection"]["rows"]
                    file_name = st.session_state["df_upload"]["文件名"].values.tolist()
                # if selected_rows:
                #     documents = [st.session_state["documents"][i] for i in selected_rows]
                #     st.session_state["select_documents"] = documents
                #     __ = [st.write(f"✅使用文档：{file_name[i]}") for i in selected_rows]

                if upload_button:
                    # st.session_state["select_documents"] = st.session_state["documents"]

                    __ = [st.write(f"✅使用文档：{name}") for name in file_name]
        elif dialogue_mode == "材料匹配":

            with st.expander("材料匹配配置", True):
                files_material_list = st.file_uploader("请上传材料清单：",
                                                       [".docx", ".xlsx", ".xls", "txt"],
                                                       accept_multiple_files=True,
                                                       )

                pkg_file = st.file_uploader("请上传能力材料：",
                                            [i for ls in PKG_DICT.values() for i in ls],
                                            accept_multiple_files=True,
                                            )
                print("files_material_list", files_material_list)
                print("pkg_file", pkg_file)
                upload_button = st.button("开始上传", disabled=(len(files_material_list) != 1 or len(pkg_file) != 1))

                if upload_button:
                    st.session_state.setdefault("documents", [])
                    st.session_state.setdefault("pkg_file", [])

                    st.session_state.setdefault("df_upload", pd.DataFrame([], columns=["文件名"]))

                    if len(files_material_list) == 1:  # 清单解析
                        doc_id, success_info, table_df, documents = upload_temp_docs_v2(files_material_list, api)
                        print("files", files_material_list)
                        print("success_info", success_info)
                        print("table_df", table_df)
                        print("documents", documents)
                        st.session_state["file_chat_id"] = doc_id
                        st.session_state["documents"] = documents
                        st.success(success_info)
                        st.session_state["df_upload"] = pd.DataFrame(table_df)

                    if len(pkg_file) == 1:  # 压缩包解析，程度为包内名称
                        doc_id, success_info, table_df, documents = upload_temp_pkgfile(pkg_file, api)
                        # documents = [[k["page_content"] for k in doc] for doc in documents]
                        print("files", files_material_list)
                        print("success_info", success_info)
                        print("table_df", table_df)
                        print("documents", documents)
                        st.session_state["pkg_file"] = documents
                        st.success(success_info)
                        if len(st.session_state["df_upload"]) != 0:
                            st.session_state["df_upload"] = pd.concat(
                                [st.session_state["df_upload"], pd.DataFrame(table_df)],
                                ignore_index=True
                            )
                    file_name = st.session_state["df_upload"]["文件名"].values.tolist()
                # if selected_rows:
                #     documents = [st.session_state["documents"][i] for i in selected_rows]
                #     st.session_state["select_documents"] = documents
                #     __ = [st.write(f"✅使用文档：{file_name[i]}") for i in selected_rows]

                if upload_button:
                    # st.session_state["select_documents"] = st.session_state["documents"]

                    __ = [st.write(f"✅使用文档：{name}") for name in file_name]
        elif dialogue_mode == "日志解析":
            with st.expander("日志解析配置", True):
                files = st.file_uploader("上传日志：",
                                         [i for ls in LOG_DICT.values() for i in ls],
                                         accept_multiple_files=True,
                                         )

                print("files", files)
                ## Bge 模型会超过1
                # score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                upload_button = st.button("开始上传", disabled=len(files) != 1)
                if upload_button:
                    st.session_state.setdefault("documents", [])
                    # st.session_state.setdefault("select_documents", [])
                    # st.session_state.setdefault("selected_rows", [])
                    #
                    st.session_state.setdefault("df_upload", pd.DataFrame([], columns=["文件名"]))

                    if len(files) != 0:
                        doc_id, success_info, table_df, documents = upload_temp_logfile(files, api)
                        # documents = [[k["page_content"] for k in doc] for doc in documents]
                        print("files", files)
                        print("success_info", success_info)
                        print("table_df", table_df)
                        print("documents", documents)
                        st.session_state["file_chat_id"] = doc_id
                        st.session_state["documents"] = documents
                        st.success(success_info)
                        st.session_state["df_upload"] = pd.DataFrame(table_df)
                    # selection = st.dataframe(st.session_state.df_upload,
                    #                          use_container_width=True,
                    #                          hide_index=True,
                    #                          on_select="rerun",
                    #                          selection_mode="multi-row",  # 允许多行选择
                    #                          key="my_dataframe",
                    #                          )
                    # selected_rows = selection["selection"]["rows"]
                    file_name = st.session_state["df_upload"]["文件名"].values.tolist()
                # if selected_rows:
                #     documents = [st.session_state["documents"][i] for i in selected_rows]
                #     st.session_state["select_documents"] = documents
                #     __ = [st.write(f"✅使用文档：{file_name[i]}") for i in selected_rows]

                if upload_button:
                    # st.session_state["select_documents"] = st.session_state["documents"]

                    __ = [st.write(f"✅使用文档：{name}") for name in file_name]

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        if parse_command(text=prompt, modal=modal):  # 用户输入自定义命令
            st.rerun()
        else:
            history = get_messages_history(history_len)
            chat_box.user_say(prompt)
            if dialogue_mode == "LLM 对话":
                chat_box.ai_say("正在思考...")
                text = ""
                message_id = ""

                print("prompt", prompt,
                      "history", history,
                      "conversation_id", conversation_id,
                      "llm_model", llm_model,
                      "prompt_template_name", prompt_template_name,
                      "temperature", temperature)

                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
                chat_box.show_feedback(**feedback_kwargs,
                                       key=message_id,
                                       on_submit=on_feedback,
                                       kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})


            elif dialogue_mode == "知识库问答":
                chat_box.ai_say([
                    f"正在查询知识库 `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
                ])
                text = ""
                for d in api.knowledge_base_chat(prompt,
                                                 knowledge_base_name=selected_kb,
                                                 top_k=kb_top_k,
                                                 score_threshold=score_threshold,
                                                 history=history,
                                                 model=llm_model,
                                                 prompt_name=prompt_template_name,
                                                 temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
            elif dialogue_mode == "文件对话":
                if st.session_state["file_chat_id"] is None:
                    st.error("请先上传文件再进行对话")
                    st.stop()
                chat_box.ai_say([
                    f"正在查询文件 `{st.session_state['file_chat_id']}` ...",
                    Markdown("...", in_expander=True, title="文件匹配结果", state="complete"),
                ])
                text = ""
                print("prompt", prompt,
                      "file_chat_id", st.session_state["file_chat_id"],
                      "select_documents", st.session_state["documents"],
                      "kb_top_k", kb_top_k,
                      "history", history,
                      "llm_model", llm_model,
                      "prompt_template_name", prompt_template_name,
                      "temperature", temperature)
                for d in api.file_chat_v2(prompt,
                                          knowledge_id=st.session_state["file_chat_id"],
                                          documents=st.session_state["documents"],
                                          top_k=kb_top_k,
                                          history=history,
                                          model_name=llm_model,
                                          prompt_name=prompt_template_name,
                                          temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

            elif dialogue_mode == "材料匹配":
                if st.session_state["file_chat_id"] is None:
                    st.error("请先上传文件再进行对话")
                    st.stop()
                chat_box.ai_say([
                    f"正在配对文件 `{st.session_state['file_chat_id']}` ...",
                    Markdown("...", in_expander=True, title="生成材料匹配报告", state="complete"),
                ])

                documents = st.session_state["documents"]
                pkg_file = st.session_state["pkg_file"]
                prompt0 = "分析【材料清单】和【待匹配清单】内容，分析内容为：有哪些文件出现在待匹配清单中，有哪些文件未出现在待匹配清单中\n"
                prompt1 = f"【材料清单】如下：{'|'.join([d['page_content'] for d in documents[0]])}\n"
                prompt2 = f"【待匹配清单】如下：{'|'.join([d for d in pkg_file[0][1]])}\n"
                prompt += prompt0 + prompt1 + prompt2
                print("prompt", prompt,
                      "history", history,
                      "conversation_id", conversation_id,
                      "llm_model", llm_model,
                      "prompt_template_name", prompt_template_name,
                      "temperature", temperature)

                text = ""
                message_id = ""
                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
                chat_box.show_feedback(**feedback_kwargs,
                                       key=message_id,
                                       on_submit=on_feedback,
                                       kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})
                chat_box.ai_say([
                    f"配对完毕 ，请点击材料匹配报告"
                ])

            elif dialogue_mode == "日志解析":
                if st.session_state["file_chat_id"] is None:
                    st.error("请先上传文件再进行对话")
                    st.stop()
                chat_box.ai_say([
                    f"正在进行日志解析 `{st.session_state['file_chat_id']}` ...",
                    Markdown("...", in_expander=True, title="生成日志解析报告", state="complete"),
                ])
                prompt_list = []
                file_name = "".join(st.session_state["df_upload"]["文件名"].values.tolist())
                if "操作日志" in file_name:
                    account_counts, daily_counts, operation_counts, account_daily_counts, error_login_counts, error_login_daily_counts = \
                        st.session_state["documents"][0]
                    prompt1 = f"【主账号操作频率分析】:{str(account_counts)},其中key为主账号名称，value为操作频率"
                    prompt2 = f"【按操作时间分类统计】:{str(daily_counts)},其中其中key为日期，value为操作频率"
                    prompt3 = f"【主账号操作内容统计】:{str(operation_counts)},其中key为操作内容，value为操作频率"
                    prompt4 = f"【按主账号和操作时间分组统计】:{str(account_daily_counts)},其中key为日期和主账号名称，value为操作频率"
                    prompt5 = f"【异常时间段主账号操作频率分析】:{str(account_daily_counts)},请分析异常时间点0~6点主账号操作频率，其中key为主账号名称，value为操作频率，如果统计结果为空，请直接返回：无异常时间段操作记录。"
                    prompt6 = f"【异常时间段主账号日期维度的操作频率分析】:{str(error_login_daily_counts)},请分析异常时间点0~6点主账号基于日期维度的操作频率，其中key为操作内容，value为操作频率，如果统计结果为空，请直接返回：无异常时间段操作记录（日期维度）。"
                    if len(prompt1) < 2048:
                        prompt_list.append(prompt1)
                    if len(prompt2) < 2048:
                        prompt_list.append(prompt2)
                    if len(prompt3) < 2048:
                        prompt_list.append(prompt3)
                    if len(prompt4) < 2048:
                        prompt_list.append(prompt4)
                    if len(prompt5) < 2048:
                        prompt_list.append(prompt5)
                    if len(prompt6) < 2048:
                        prompt_list.append(prompt6)
                if "登录日志" in file_name:
                    login_counts, daily_counts, pivot_table, error_login_counts, error_pivot_table = st.session_state[
                        "documents"][0]

                    prompt1 = f"【ID登录频率分析】:{str(login_counts)},其中key为登录ID名称，value为操作频率"
                    prompt2 = f"【按登录时间分类统计】:{str(daily_counts)},其中key为日期，value为操作频率"
                    prompt3 = f"【按登录ID和日期分组记录数量】:{str(pivot_table)},其中key为日期和登录ID，value为操作频率"
                    prompt4 = f"【异常时间段主账号登录频率分析】:{str(error_login_counts)},请分析异常时间点0~6点主账号登录频率,其中key为日期，value为操作频率，如果统计结果为空，请直接返回：无异常时间段登录记录。"
                    prompt5 = f"【异常时间段主账号日期维度的登录频率分析】:{str(error_pivot_table)},请分析异常时间点0~6点主账号基于日期维度的登录频率,key为日期和主账号名称，value为操作频率，如果统计结果为空，请直接返回：无异常时间段登录记录（日期维度）。"
                    if len(prompt1) < 2048:
                        prompt_list.append(prompt1)
                    if len(prompt2) < 2048:
                        prompt_list.append(prompt2)
                    if len(prompt3) < 2048:
                        prompt_list.append(prompt3)
                    if len(prompt4) < 2048:
                        prompt_list.append(prompt4)
                    if len(prompt5) < 2048:
                        prompt_list.append(prompt5)

                if "使用日志" in file_name:
                    print("st.session_state['documents']", st.session_state["documents"])
                    login_id_counts, operation_counts, daily_counts, login_operation_counts, login_daily_counts, error_login_counts, error_login_daily_counts = \
                        st.session_state["documents"][0]

                    prompt1 = f"【登录ID使用频率统计】:{str(login_id_counts).strip().replace(' ', '')},其中key为登录ID，value为使用频率"
                    prompt2 = f"【操作内容使用频率统计】:{str(operation_counts).strip().replace(' ', '')},其中key为操作内容，value为使用频率"
                    prompt3 = f"【按访问时间分类统计】:{str(daily_counts).strip().replace(' ', '')},其中key为访问时间，value为使用频率"
                    prompt4 = f"【按登录ID和操作内容分组统计】:{str(login_operation_counts).strip().replace(' ', '')},其中key为操作内容和登录ID，value为使用频率"
                    prompt5 = f"【按登录ID和日期分组统计】:{str(login_daily_counts).strip().replace(' ', '')},其中key为操作日期和登录ID，value为使用频率"
                    prompt6 = f"【异常时间段主账号使用频率分析】:{str(error_login_counts).strip().replace(' ', '')},请分析异常时间点0~6点主账号使用频率,其中key为登录ID名称，value为操作频率，如果统计结果为空，请直接返回：无异常时间段使用记录"
                    prompt7 = f"【异常时间段主账号日期维度的使用频率分析】:{str(error_login_daily_counts).strip().replace(' ', '')},请分析异常时间点0~6点主账号基于日期维度的使用频率,其中key为日期和登录ID，value为登录频率，如果统计结果为空，请直接返回：无异常时间段使用记录（日期维度）。"
                    if len(prompt1) < 2048:
                        prompt_list.append(prompt1)
                    if len(prompt2) < 2048:
                        prompt_list.append(prompt2)
                    if len(prompt3) < 2048:
                        prompt_list.append(prompt3)
                    if len(prompt4) < 2048:
                        prompt_list.append(prompt4)
                    if len(prompt5) < 2048:
                        prompt_list.append(prompt5)
                    if len(prompt6) < 2048:
                        prompt_list.append(prompt6)
                    if len(prompt7) < 2048:
                        prompt_list.append(prompt7)

                # exit()
                prompt += "|".join(prompt_list)
                print("len_prompt", len(prompt))

                print("prompt", prompt,
                      "history", history,
                      "conversation_id", conversation_id,
                      "llm_model", llm_model,
                      "prompt_template_name", prompt_template_name,
                      "temperature", temperature)

                text = ""
                message_id = ""
                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature
                                  )
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
                chat_box.show_feedback(**feedback_kwargs,
                                       key=message_id,
                                       on_submit=on_feedback,
                                       kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})
                chat_box.ai_say([
                    f"解析完毕 ，请点击日志分析报告"
                ])

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
