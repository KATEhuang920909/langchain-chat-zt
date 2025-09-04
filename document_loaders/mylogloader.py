import streamlit as st
import pandas as pd
import os

class Preprocess():
    def __int__(self, ):
        # self.chunk = KnowledgeChunk()
        self.documents = []
        self.df = {}
        self.idf = {}
        self.average_doc_len = 0

    def get_bm25_model(self, contents):
        # BM25_store
        _ = [self.add_document(doc) for doc in contents]
        self.calculate_idf()

        # Store in session

    # 通用处理
    def process_documents(self, uploaded_files):
        # if st.session_state.documents_loaded:
        #     return

        # # Text splitting
        # print(documents)
        # exit()
        documents, doc_name = self.file_parse(uploaded_files)  # 解析自定义材料
        # print("documents", documents)
        # print("doc_name", doc_name)
        for doc in documents:
            text_contents = self.split_sentences(doc[0].page_content)  # 切片
            st.session_state.documents += text_contents
        # print(st.session_state.documents)

        self.get_bm25_model(st.session_state.documents)
        st.session_state.retrieval_pipeline = {"texts": st.session_state.documents}
        st.write(f"🔗 Total Chunk: {len(st.session_state.documents)}")

        st.session_state.documents_loaded = True
        # st.session_state.processing = False
        return documents, doc_name

    def file_parse(self, uploaded_files):
        files_bag, file_name = [], []
        # st.session_state.processing = True
        # documents = []

        # Create temp directory
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Process files
        for file in uploaded_files:
            try:
                file_path = os.path.join("temp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())


                documents = loader.load()
                files_bag.append(documents)
                file_name.append(file.name)
                os.remove(file_path)

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                return
        return files_bag, file_name



    # 日志材料处理
    def log_parse(self, uploaded_files):
        # st.session_state.processing = True
        # documents = []

        # Create temp directory
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Process files
        for file in uploaded_files:
            try:
                file_path = os.path.join("temp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                if file.name.endswith(".xlsx") or file.name.endswith(".xls"):
                    documents = pd.read_excel(file_path)
                elif file.name.endswith(".csv"):
                    documents = pd.read_csv(file_path)
                else:
                    continue

                os.remove(file_path)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                return

        if "操作日志" in uploaded_files[0].name:  # 主机操作日志
            account_counts, daily_counts, operation_counts, account_daily_counts, error_login_counts, error_login_daily_counts = OperationLog(
                documents)
            return "操作日志", [account_counts, daily_counts, operation_counts, account_daily_counts,
                                error_login_counts, error_login_daily_counts]
        elif "登录日志" in uploaded_files[0].name:  # 主机登录日志
            login_counts, daily_counts, pivot_table, error_login_counts, error_pivot_table = EnterLog(documents)
            return "登录日志", [login_counts, daily_counts, pivot_table, error_login_counts, error_pivot_table]
        elif "使用日志" in uploaded_files[0].name:  # 主机登录日志
            login_id_counts, operation_counts, daily_counts, login_operation_counts, login_daily_counts, error_login_counts, error_login_daily_counts = UseLog(
                documents)
            return "使用日志", [login_id_counts, operation_counts, daily_counts, login_operation_counts,
                                login_daily_counts, error_login_counts, error_login_daily_counts]

        # def process_log(self, uploaded_files):
        #     # if st.session_state.documents_loaded:
        #     #     return
        #
        #     # # Text splitting
        #     # print(documents)
        #     # exit()
        #     log_files = self.log_parse(uploaded_files)
        #     if "操作日志" in  uploaded_files[0].name:#主机操作日志
        #         account_counts, daily_counts, operation_counts, account_daily_counts =OperationLog(log_files)

        # st.session_state.processing = False


if __name__ == '__main__':
    process = Preprocess()
    # chunk = process.split_recursive(
    #     "近日，中共中央办公厅印发了《全国党员教育培训工作规划（2024—2028年）》，并发出通知，要求各地区各部门结合实际认真贯彻落实。《全国党员教育培训工作规划（2024—2028年）》全文如下。为加强和改进党员教育培训工作，锻造过硬党员队伍，不断增强党的创造力、凝聚力、战斗力，根据《中国共产党章程》和《中国共产党党员教育管理工作条例》等党内法规，制定本规划。一、总体要求坚持以习近平新时代中国特色社会主义思想为指导，深入贯彻党的二十大和二十届二中、三中全会精神，全面贯彻习近平总书记关于党的建设的重要思想、关于党的自我革命的重要思想，深入落实新时代党的建设总要求和新时代党的组织路线，以用党的创新理论武装全党为首要政治任务，以增强党性、提高素质、发挥作用为重点，坚持政治引领、分类指导、守正创新、服务大局，教育引导全体党员深刻领悟“两个确立”的决定性意义，增强“四个意识”、坚定“四个自信”、做到“两个维护”，开拓进取、干事创业，为以中国式现代化全面推进强国建设、民族复兴伟业提供思想政治保证和能力支撑。",
    #     chunk_size=20)
    # print(chunk)
    # _ = [process.add_document(doc) for doc in chunk]
    # # process.add_document(chunk)
    # process.calculate_idf()
    # print(process.search("近日，中共中央办公厅印发了什么"))
    loader = Docx2txtLoader(r"D:\work\中台\中台安全运营工具\test\中台检查文档需求清单.docx")
    documents = loader.load()
    print(documents)
