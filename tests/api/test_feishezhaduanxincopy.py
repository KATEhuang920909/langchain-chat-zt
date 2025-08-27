import os
import re
import requests
import time

# API base URL
api_base_url = "http://localhost:19248"

# Request headers
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# Define test_knowledge_chat function
def test_llm_chat(query, data):
    """
    This function sends a POST request with a query to the knowledge chat API.
    """
    url = f"{api_base_url}/chat/chat"
    
    data["query"] = query  # Update the query in the request data
    response = requests.post(url, headers=headers, json=data)

    # Extract "answer" field from the response text
    match = re.search(r'"text":\s*"([^"]*[\u4e00-\u9fff]+[^"]*)"', response.text)
    
    if match:
        # Extract and return the "answer" field's content
        answer = match.group(1)
        return answer
    else:
        return None

def process_text_file(file_path, output_file):
    """
    This function processes the text file, extracts data, and calls the API.
    It computes the "是" response ratio and saves the results to a text file.
    """
    total_rows = 0
    total_yes_count = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        yes_count = 0
        row_count = 0

        # Read from the text file
        with open(file_path, 'r', encoding='utf-8') as txtfile:
            lines = txtfile.readlines()

            # Process each line in the text file
            for line in lines:
                full_message = line.strip()  # Remove leading/trailing whitespace
                
                # Debug: Print extracted message to verify
                # print(f"Extracted message: {full_message}")

                # Prepare the query
                query = f"【{full_message}】是诈骗短信吗?"

                # Prepare data for API call
                data = {
                    "query": query,
                    "history": [
                        {
                            "role": "user",
                            "content": "你好"
                        },
                        {
                            "role": "assistant",
                            "content": "你好，我是人工智能大模型"
                        }
                    ],
                    "stream": False,
                    "model_name": "测试模型1-14B",
                    "temperature": 0.01,
                    "max_tokens": 0,
                    "prompt_name": "基础LLM对话测试"
                }

                # Call the knowledge chat API
                answer = test_llm_chat(query, data)
                print(query)
                print(answer)

                # Check if the answer is "是"
                if "是" in answer:
                    yes_count += 1
                    print("yes_count:" + str(yes_count))

                row_count += 1
                print("row_count:" + str(row_count))

        # Calculate and save the ratio for the processed file
        if row_count > 0:
            # yes_ratio = yes_count / row_count
            # outfile.write(f"Total tested: {row_count}, '是': {yes_count}, '是' Ratio: {yes_ratio:.2%}\n")
            # print(f"Total tested: {row_count}, '是': {yes_count}, '是' Ratio: {yes_ratio:.2%}")

            # Update total counters
            total_rows += row_count
            total_yes_count += yes_count

        # Calculate and save the overall ratio
        if total_rows > 0:
            total_yes_ratio = total_yes_count / total_rows
            outfile.write(f"Total tested: {total_rows}, '是': {total_yes_count}, Total '是' Ratio: {total_yes_ratio:.2%}\n")
            print(f"Total tested: {total_rows}, '是': {total_yes_count}, Total '是' Ratio: {total_yes_ratio:.2%}")
        else:
            outfile.write("No valid rows processed.\n")
            print("No valid rows processed.")

if __name__ == "__main__":
    start_time = time.time()  # 开始计时

    file_path = "/home/ding/LLM/generated_messages.txt"  # Replace with your text file path
    output_file = "llmceshizhengyangben.txt"  # Output text file to save the results

    process_text_file(file_path, output_file)

    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time  # 计算总时间
    elapsed_time_str = f"Total execution time: {elapsed_time:.2f} seconds\n"  # 格式化时间

    print(elapsed_time_str)  # 打印执行时间

    # 将执行时间写入输出文件
    with open(output_file, 'a', encoding='utf-8') as outfile:
        outfile.write(elapsed_time_str)
