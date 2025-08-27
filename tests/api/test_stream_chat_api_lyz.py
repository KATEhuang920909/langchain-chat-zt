import requests
import json
import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs import BING_SUBSCRIPTION_KEY
from server.utils import api_address

from pprint import pprint


# api_base_url = api_address()
api_base_url = "http://localhost:19248"


def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "="*30)
    pprint(d)


def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "="*30)
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}
data = {
    "query": "【【360借条】喜迎双11，全民免息狂欢，送您最 长12期 免息 福利，再送您借 款额 度免 息用最长15天资格> https://s.】是诈骗短信吗?",
    "knowledge_base_name": "fujian",
    "top_k": 3,
    "score_threshold": 1,
    "history": [
        {
            "role": "user",
            "content": "你好"
        },
        {
            "role": "assistant",
            "content": "你好，我是 ChatGLM"
        }
    ],
    "stream": False,
    "model_name": "测试模型1-14B",
    "temperature": 0.01,
    "max_tokens": 0,        
    "prompt_name": "反诈知识问答加固"
}
# data = {
#     "query": "请用100字左右的文字介绍自己",
#     "history": [
#         {
#             "role": "user",
#             "content": "你好"
#         },
#         {
#             "role": "assistant",
#             "content": "你好，我是人工智能大模型"
#         }
#     ],
#     "stream": False,
#     "model_name": "测试模型1-14B",
#     "temperature": 0.01,
#     "max_tokens": 0,
#     "prompt_name": "反诈对话"
# }


# def test_chat_chat(api="/chat/chat"):
#     url = f"{api_base_url}{api}"
#     print(url)
#     dump_input(data, api)
#     response = requests.post(url, headers=headers, json=data, stream=True)
#     dump_output(response, api)
#     print(response)
    # assert response.status_code == 200


def test_knowledge_chat(api="/chat/knowledge_base_chat"):
    url = f"{api_base_url}{api}"

    response = requests.post(url, headers=headers, json=data)

    # dump_input(data, api)
    # dump_output(response, api)
    # print()
    # # print(response)
    # print("Response Text:", response.text)

    # if response.status_code == 200:
    #     # 将响应数据解析为JSON
    #     response_data = response.json()

    #     # 提取 answer 字段并打印
    #     answer = response_data.get("answer", "No answer found.")
    #     print(answer)
    # else:
    #     print(f"请求失败，状态码: {response.status_code}")
    match = re.search(r'"answer":\s*"([^"]*[\u4e00-\u9fff]+[^"]*)"', response.text)
    
    if match:
        # 提取并打印出 "answer" 字段的内容
        answer = match.group(1)
        print(answer)
    else:
        print("未找到符合条件的回答")


def test_search_engine_chat(api="/chat/search_engine_chat"):
    global data

    data["query"] = "室温超导最新进展是什么样？"

    url = f"{api_base_url}{api}"
    for se in ["bing", "duckduckgo"]:
        data["search_engine_name"] = se
        dump_input(data, api + f" by {se}")
        response = requests.post(url, json=data, stream=True)
        if se == "bing" and not BING_SUBSCRIPTION_KEY:
            data = response.json()
            assert data["code"] == 404
            assert data["msg"] == f"要使用Bing搜索引擎，需要设置 `BING_SUBSCRIPTION_KEY`"

        print("\n")
        print("=" * 30 + api + f" by {se}  output" + "="*30)
        for line in response.iter_content(None, decode_unicode=True):
            data = json.loads(line[6:])
            if "answer" in data:
                print(data["answer"], end="", flush=True)
        assert "docs" in data and len(data["docs"]) > 0
        pprint(data["docs"])
        assert response.status_code == 200

test_knowledge_chat()
# test_chat_chat()