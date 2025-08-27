# 批量使用kimi API测试数据
import os
import csv
import random
import time
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-nZdhGtqxMQwwngzaOOptG2ICIS9SSNaguKNPrMvshiYGa5HT",
    base_url="https://api.moonshot.cn/v1"
)

# 过滤器
donefile = open("/home/ding/LLM/baseline/result/result_0.txt", "r", encoding='utf-8')
donefilelist = []
for line in donefile.readlines():
    existfilename = line[3:21]
    donefilelist.append(existfilename)


# 函数：使用 KimI API 提问并获取回答
def ask_kimi(question_prompt):
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "User: " + question_prompt},
        ]
    )
    response = completion.choices[0].message.content
    return response

# 函数：处理 CSV 文件并查询每条信息是否为诈骗短信
def process_csv_files(folder_path, output_file):
    total_yes_count = 0
    total_rows = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in os.listdir(folder_path):
            if file_name in donefilelist:
                a = 1
            else:
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                if file_path.endswith('.csv'):
                    file_yes_count = 0  # 初始化该文件的涉诈短信计数
                    file_rows = 0  # 初始化该文件的行计数

                    with open(file_path, 'r', encoding='utf-8') as csvfile:
                        reader = list(csv.reader(csvfile))
                        selected_rows = random.sample(reader, min(50, len(reader)))

                        for row in selected_rows:
                            full_message = ' '.join(row).strip()
                            question_prompt = f"【{full_message}】是诈骗短信吗?"
                            prompt = (
                                '根据你的知识，简洁和专业的来回答问题。如果你能肯定判断是诈骗短信，请说“是”。如果你不能肯定判断是涉诈短信，请说“我无法肯定这条短信为涉诈短信”，'
                                '不允许在答案中添加编造成分，答案请使用中文。\n'
                                '请严格按照规定进行输出，你的输出只能是：“是”, 或者 “我无法肯定这条短信为涉诈短信” ，不要出现任何其他不同的回答或字眼或符号\n'
                            )
                            response = ask_kimi(question_prompt + prompt)
                            print()
                            print(question_prompt + prompt)
                            print(response)
                            time.sleep(30)  # 每次请求后暂停30秒

                            file_rows += 1
                            if "是" in response:
                                file_yes_count += 1

                    # 记录每个文件的结果
                    file_yes_ratio = file_yes_count / file_rows if file_rows > 0 else 0
                    outfile.write(f"\n文件：{file_name} 处理了 {file_rows} 条记录，其中 {file_yes_count} 条为诈骗短信，占比：{file_yes_ratio:.2f}\n")
                    print(f"文件：{file_name} 处理了 {file_rows} 条记录，其中 {file_yes_count} 条为诈骗短信，占比：{file_yes_ratio:.2f}\n")

                    # 累加到总计数
                    total_yes_count += file_yes_count
                    total_rows += file_rows

        # 输出总体结果
        total_yes_ratio = total_yes_count / total_rows if total_rows > 0 else 0
        outfile.write(f"\n总计：处理了 {total_rows} 条记录，其中 {total_yes_count} 条为诈骗短信，占比：{total_yes_ratio:.2f}\n")
        print(f"总计：处理了 {total_rows} 条记录，其中 {total_yes_count} 条为诈骗短信，占比：{total_yes_ratio:.2f}\n")


folder_path = "/home/ding/LLM/Processed_CSVs"
output_file = "/home/ding/LLM/baseline/result/result.txt"
process_csv_files(folder_path, output_file)
