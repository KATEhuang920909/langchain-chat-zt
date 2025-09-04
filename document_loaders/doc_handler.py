import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
from collections import defaultdict
#
#
# class Preprocess:
#     def __init__(self, file_path=None):
#         # self.chunk = KnowledgeChunk()
#         pass
#
#     def file_parse(self, file_path):
#         files_bag, file_name = [], []
#         # Process files
#         try:
#
#             loader = Docx2txtLoader(file_path)
#             documents = loader.load()
#
#         except Exception as e:
#             st.error(f"Error processing {file_path}: {str(e)}")
#             return
#         return documents
#
#         # def process_log(self, uploaded_files):
#         #     # if st.session_state.documents_loaded:
#         #     #     return
#         #
#         #     # # Text splitting
#         #     # print(documents)
#         #     # exit()
#         #     log_files = self.log_parse(uploaded_files)
#         #     if "操作日志" in  uploaded_files[0].name:#主机操作日志
#         #         account_counts, daily_counts, operation_counts, account_daily_counts =OperationLog(log_files)
#
#         # st.session_state.processing = False
#
if __name__ == '__main__':
    loader = Docx2txtLoader(file_path = r"D:\work\中台\中台安全运营工具\test\中台检查文档需求清单.docx")

    documents = loader.load()
    print(documents)
