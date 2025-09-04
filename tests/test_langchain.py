from langchain_community.document_loaders import UnstructuredExcelLoader

# 指定你的 Excel 文件路径
file_path = r"D:\work\中台\中台安全运营工具\test\中台检查文档需求清单@充电桩场景精细化选址能力v1\系统安全\1-1 日志：能力使用日志.xlsx"  # 请替换为你的文件路径

# 初始化加载器
# mode 参数可选 "single" (返回一个文档) 或 "elements" (返回结构化元素)
loader = UnstructuredExcelLoader(file_path, mode="single")

# 加载文档
docs = loader.load()

# 查看加载的文档数量
print(f"Loaded {len(docs)} documents.")

# 遍历并打印每个文档的内容和元数据（示例）
for doc in docs:
    print(f"Content: {doc.page_content[:200]}...")  # 打印前200个字符
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)


