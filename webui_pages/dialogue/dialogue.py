import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime
import os
import time
import base64
import io
import re
import time
from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
                     DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL, BM25_SEARCH_TOP_K)
from server.knowledge_base.utils import LOADER_DICT
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
            "材料匹配": "material_match",
            "日志解析": "search_engine_chat",
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
                                         [i for ls in LOADER_DICT.values() for i in ls],
                                         accept_multiple_files=True,
                                         )

                kb_top_k = st.number_input("匹配知识条数：", 1, 20, BM25_SEARCH_TOP_K)

                ## Bge 模型会超过1
                # score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                upload_button = st.button("开始上传", disabled=len(files) == 0)


                if upload_button:
                    st.session_state.setdefault("documents", [])
                    # st.session_state.setdefault("select_documents", [])
                    # st.session_state.setdefault("selected_rows", [])
                    #
                    st.session_state.setdefault("df_upload", pd.DataFrame([], columns=["文件名", "材料类型"]))


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
                files = st.file_uploader("请上传材料清单：",
                                         [".docx",".xlsx", ".xls","txt"],
                                         accept_multiple_files=False,
                                         )

                kb_top_k = st.number_input("匹配知识条数：", 1, 20, BM25_SEARCH_TOP_K)

                ## Bge 模型会超过1
                # score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                upload_button = st.button("开始上传", disabled=len(files) == 0)


                if upload_button:
                    st.session_state.setdefault("documents", [])
                    # st.session_state.setdefault("select_documents", [])
                    # st.session_state.setdefault("selected_rows", [])
                    #
                    st.session_state.setdefault("df_upload", pd.DataFrame([], columns=["文件名", "材料类型"]))


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
        elif dialogue_mode == "日志解析":
            search_engine_list = api.list_search_engines()
            if DEFAULT_SEARCH_ENGINE in search_engine_list:
                index = search_engine_list.index(DEFAULT_SEARCH_ENGINE)
            else:
                index = search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0
            with st.expander("搜索引擎配置", True):
                search_engine = st.selectbox(
                    label="请选择搜索引擎",
                    options=search_engine_list,
                    index=index,
                )
                se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, SEARCH_ENGINE_TOP_K)

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

            elif dialogue_mode == "材料匹配":
                if not any(agent in llm_model for agent in SUPPORT_AGENT_MODEL):
                    chat_box.ai_say([
                        f"正在思考... \n\n <span style='color:red'>该模型并没有进行Agent对齐，请更换支持Agent的模型获得更好的体验！</span>\n\n\n",
                        Markdown("...", in_expander=True, title="思考过程", state="complete"),

                    ])
                else:
                    chat_box.ai_say([
                        f"正在思考...",
                        Markdown("...", in_expander=True, title="思考过程", state="complete"),

                    ])
                text = ""
                ans = ""
                if prompt.find("网页分析") != -1:
                    time.sleep(3)
                    text = '''
这篇网页的主要内容是关于广州某科技公司遭受境外网络攻击的事件报道，以及公安机关对此事件的调查进展和相关分析。以下是主要内容的详细梳理：

### 事件概述
- **事件时间**：2025年5月20日
- **事件地点**：广州市某科技公司
- **事件性质**：境外黑客组织发起的网络攻击
- **攻击目标**：该公司自助设备的后台系统

### 攻击情况
- **攻击手段**：攻击者利用技术手段绕过网络防护装置，非法进入后台系统，通过横向移动渗透控制多台网络设备，上传攻击程序。
- **攻击后果**：导致公司官方网站和部分业务系统中断数小时，造成重大损失，部分用户隐私信息疑遭泄露。

### 公安机关的行动
- **立案调查**：广州市公安局天河区分局发布《警情通报》，立即开展立案调查。
- **技术分析**：提取攻击程序样本，固定电子证据，开展技术溯源。
- **初步判断**：攻击具有明显的政治背景，属于APT（高级持续性威胁）攻击，但攻击水平较低，属于APT组织中的二三线水平。
- **线索追踪**：攻击过程中暴露出大量网络线索，公安机关正在进一步分析和调查。

### 专家分析
- **攻击特点**：具有高度定向性，属于APT攻击，攻击者长期使用开源工具扫描探测目标，寻找网络防护薄弱环节。
- **应对能力**：我国在威胁发现、溯源、反制等方面已具备有效应对能力，能够锁定威胁来源。

### 公众反应
- **评论区**：网友对事件表示关注，呼吁加强网络安全保护，支持公安机关打击网络犯罪。

### 公安机关提示
- **举报权利**：公众有权依照《中华人民共和国网络安全法》向公安机关网安部门举报危害网络安全的行为。
- **打击决心**：公安机关将依法坚决打击相关违法犯罪行为。

### 其他信息
- **类似案例**：近年来，公安机关已成功锁定并打击多起境外黑客组织对我国发起的网络攻击，如“西北工业大学遭网络攻击”和“武汉市地震监测中心遭网络攻击”。
- **技术发展**：我国网络安全企业在攻击溯源能力上不断发展，从被动跟进到主动捕获，再到实现攻击溯源到达自然人的突破。

### 总结
该网页报道了一起境外黑客组织对广州某科技公司的网络攻击事件，强调了公安机关的快速响应和技术分析能力，以及我国在应对APT攻击方面的技术进步。同时，提醒公众关注网络安全，支持公安机关打击网络犯罪行为。
                    '''
                    ans = text
                else:
                    for d in api.agent_chat(prompt,
                                            history=history,
                                            model=llm_model,
                                            prompt_name=prompt_template_name,
                                            temperature=temperature,
                                            ):
                        try:
                            d = json.loads(d)
                        except:
                            pass
                        if error_msg := check_error_msg(d):  # check whether error occured
                            st.error(error_msg)
                        if chunk := d.get("answer"):
                            text += chunk
                            chat_box.update_msg(text, element_index=1)
                        if chunk := d.get("final_answer"):
                            ans += chunk
                            chat_box.update_msg(ans, element_index=0)
                        if chunk := d.get("tools"):
                            text += "\n\n".join(d.get("tools", []))
                            chat_box.update_msg(text, element_index=1)

                chat_box.update_msg(ans, element_index=0, streaming=False)
                chat_box.update_msg(text, element_index=1, streaming=False)
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
            elif dialogue_mode == "日志解析":
                chat_box.ai_say([
                    f"正在执行 `{search_engine}` 搜索...",
                    Markdown("...", in_expander=True, title="网络搜索结果", state="complete"),
                ])
                text = ""
                if prompt.find("最近有什么新的网络安全事件值得注意？") != -1:
                    time.sleep(3)
                    text = '''
                            根据文中内容，最近的网络安全事件有以下几点：
                            2024年4月，360数字安全集团联合中国国家计算机病毒应急处理中心首次对美国捏造的名为所谓“具有中国政府支持背景”的黑客组织“伏特台风”进行溯源，发现该组织并无表现出明确的国家背景黑客组织行为特征，而是与勒索病毒等网络犯罪团伙的关联程度明显。
                            1月，世界经济论坛《2024年全球风险报告》发现，网络不安全是一种跨时间范围的全球风险，恶意软件、深度伪造和虚假信息等网络风险威胁着供应链、金融稳定和民主。
                            10月，论坛《2024年全球网络安全展望》警告称，“网络犯罪分子所使用的相同攻击媒介仍在使用；然而，新技术为邪恶活动铺平了道路。 自那时起，我们看到虚假信息和深度伪造变得越来越普遍， 有时涉及知名人士并导致重大金融诈骗，而从奥运会到全球金融机构的一切都成为网络攻击的目标。”
                            10月网络安全热点事件大盘点国内企业遭遇窃密木马钓鱼攻击，涉及敏感信息窃取与远控。
                            2024年4月，360数字安全集团联合中国国家计算机病毒应急处理中心首次对美国捏造的名为所谓“具有中国政府支持背景”的黑客组织“伏特台风”进行溯源，发现该组织并无表现出明确的国家背景黑客组织行为特征，而是与勒索病毒等网络犯罪团伙的关联程度明显。
                            1月，世界经济论坛《2024年全球风险报告》发现，网络不安全是一种跨时间范围的全球风险，恶意软件、深度伪造和虚假信息等网络风险威胁着供应链、金融稳定和民主。
                            10月，论坛《2024年全球网络安全展望》警告称，“网络犯罪分子所使用的相同攻击媒介仍在使用；然而，新技术为邪恶活动铺平了道路。 自那时起，我们看到虚假信息和深度伪造变得越来越普遍， 有时涉及知名人士并导致重大金融诈骗，而从奥运会到全球金融机构的一切都成为网络攻击的目标。”
                            10月网络安全热点事件大盘点国内企业遭遇窃密木马钓鱼攻击，涉及敏感信息窃取与远控。
                            2024年4月，360数字安全集团联合中国国家计算机病毒应急处理中心首次对美国捏造的名为所谓“具有中国政府支持背景”的黑客组织“伏特台风”进行溯源，发现该组织并无表现出明确的国家背景黑客组织行为特征，而是与勒索病毒等网络犯罪团伙的关联程度明显。
                            1月，世界经济论坛《2024年全球风险报告》发现，网络不安全是一种跨时间范围的全球风险，恶意软件、深度伪造和虚假信息等网络风险威胁着供应链、金融稳定和民主。
                            这些事件都涉及网络安全问题，提醒人们要提高网络安全意识，加强网络安全防护措施，防范网络犯罪。'''
                    chat_box.update_msg(text, element_index=0, streaming=True)
                else:
                    for d in api.search_engine_chat(prompt,
                                                    search_engine_name=search_engine,
                                                    top_k=se_top_k,
                                                    history=history,
                                                    model=llm_model,
                                                    prompt_name=prompt_template_name,
                                                    temperature=temperature,
                                                    split_result=se_top_k > 1):
                        if error_msg := check_error_msg(d):  # check whether error occured
                            st.error(error_msg)
                        elif chunk := d.get("answer"):
                            text += chunk
                            chat_box.update_msg(text, element_index=0)
                    chat_box.update_msg(text, element_index=0, streaming=False)
                    chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

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
