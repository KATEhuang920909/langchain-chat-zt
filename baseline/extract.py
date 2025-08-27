file = open("/home/ding/LLM/baseline/result/result_0.txt", "r", encoding='utf-8')

num = 0

for line in file.readlines():
    num = num + int(line[36:38])

print(num)
print(str(num/3000))