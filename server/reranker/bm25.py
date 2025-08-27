
import jieba
import math
from collections import defaultdict


class BM25Chinese:
    def __init__(self):
        self.documents = []
        self.inverted_index = defaultdict(set)
        self.df = {}
        self.idf = {}
        self.average_doc_len = 0

    def add_document(self, document):
        # 使用jieba进行中文分词
        words = list(jieba.cut(document))
        self.documents.append(words)
        doc_id = len(self.documents) - 1
        unique_words = set(words)
        for word in unique_words:
            self.inverted_index[word].add(doc_id)

    def calculate_idf(self):
        N = len(self.documents)
        self.df = {word: len(doc_ids) for word, doc_ids in self.inverted_index.items()}
        self.idf = {word: math.log((N - self.df[word] + 0.5) / (self.df[word] + 0.5)) for word in self.inverted_index}
        total_length = sum(len(doc) for doc in self.documents)
        self.average_doc_len = total_length / N

    def bm25_score(self, doc_id, query_terms, k1=1.5, b=0.75):
        doc_len = len(self.documents[doc_id])
        score = 0
        for term in query_terms:
            if term in self.documents[doc_id]:
                f = self.documents[doc_id].count(term)
                score += self.idf.get(term, 0) * (
                            (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / self.average_doc_len))))
        return score

    def search(self, query):
        query_terms = list(jieba.cut(query))
        scores = {doc_id: self.bm25_score(doc_id, query_terms) for doc_id in range(len(self.documents))}
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# if __name__ == '__main__':
#     from utils.data_preprocess import KnowledgeChunk
#
#     knowledgechunk = KnowledgeChunk()
#     # 示例用法
#     # zh_sentences = split_by_token(text, model_name="gpt-4", chunk_size=2048)
#     # 固定长度
#     # zh_sentences = split_fixed_length(text, chunk_size=20, chunk_overlap=5)
#     # 按句子切分
#     # en_sentences = split_sentences(english_text, lang='en')
#
#     # 按特殊符号切分
#     # md_chunks = split_markdown(markdown_text, heading_level=2)
#     # 语义切分
#     # semantic_chunks = split_semantic(text, threshold=0.8)
#     # 循环切分
#     # chunks = split_recursive(text, separators=["\n## ", "\n### ", "\n"])
#     # print(zh_sentences)
#     knowledge_base = open("../datasets/Knowledge_Base.txt", "r", encoding="utf8").readlines()[0]
#     corpus = knowledgechunk.split_sentences(knowledge_base, lang='zh')
#     # 示例中文文档集和查询
#
#     # 使用 jieba 进行分词
#
#     # 初始化 BM25 模型
#     # bm25 = BM25()
#     bm25 = BM25Chinese()
#     _ = [bm25.add_document(doc) for doc in corpus]
#     bm25.calculate_idf()
#     while True:
#         query = input(" please input a text")
#         # query = "测试文档"
#         results = bm25.search(query)
#         # print(results)  # 打印出搜索结果，包括文档ID和得分
#         # 输出得分
#         for idx_score in results[:10]:
#             print(f"{corpus[idx_score[0]]}：{idx_score[1]}")
