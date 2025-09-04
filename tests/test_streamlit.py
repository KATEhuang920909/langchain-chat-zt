import streamlit as st
import zipfile
import os
from io import BytesIO

# 设置页面标题
st.title("ZIP压缩包上传与解析")

# 上传ZIP文件
uploaded_file = st.file_uploader("请上传ZIP压缩包", type=["zip"])

if uploaded_file is not None:
    # 读取ZIP文件
    print(uploaded_file)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        # 获取ZIP文件中的文件列表
        file_list = zip_ref.namelist()

        # 显示压缩包内的文件信息
        st.write("压缩包内的文件列表:")
        for file_name in file_list:
            # 获取文件信息
            file_info = zip_ref.getinfo(file_name)
            # 显示文件名称和大小
            st.write(f"文件名称: {file_name}, 文件大小: {file_info.file_size} 字节")

        # 如果需要解压文件到某个路径，可以使用以下代码
        # 例如：解压到当前目录下的 'extracted_files' 文件夹
        extract_path = "extracted_files"
        if st.button("解压文件"):
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            zip_ref.extractall(extract_path)
            st.write(f"文件已解压到: {os.path.abspath(extract_path)}")
else:
    st.info("请上传一个ZIP文件")

# import streamlit as st
# import zipfile
# import os
# import shutil
#
# # 设置页面标题
# st.title("ZIP压缩包上传与解析")
#
# # 缓存路径
# CACHE_DIR = "cache"
# ZIP_CACHE_PATH = os.path.join(CACHE_DIR, "uploaded.zip")
#
# # 确保缓存目录存在
# if not os.path.exists(CACHE_DIR):
#     os.makedirs(CACHE_DIR)
#
# # 上传ZIP文件
# uploaded_file = st.file_uploader("请上传ZIP压缩包", type=["zip"])
#
# if uploaded_file is not None:
#     # 将上传的文件保存到缓存路径
#
#     with open(ZIP_CACHE_PATH, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     st.success(f"ZIP文件已缓存到: {os.path.abspath(ZIP_CACHE_PATH)}")
#
#     # 读取并解析缓存的ZIP文件
#     try:
#         with zipfile.ZipFile(ZIP_CACHE_PATH, 'r') as zip_ref:
#             # 获取ZIP文件中的文件列表
#             file_list = zip_ref.namelist()
#
#             # 显示压缩包内的文件信息
#             st.write("压缩包内的文件列表:")
#             for file_name in file_list:
#                 # 获取文件信息
#                 file_info = zip_ref.getinfo(file_name)
#                 # 显示文件名称和大小
#                 st.write(f"文件名称: {file_name}, 文件大小: {file_info.file_size} 字节")
#
#             # 如果需要解压文件到某个路径，可以使用以下代码
#             # 例如：解压到当前目录下的 'extracted_files' 文件夹
#             extract_path = "extracted_files"
#             if st.button("解压文件"):
#                 if os.path.exists(extract_path):
#                     shutil.rmtree(extract_path)  # 删除已存在的解压目录
#                 os.makedirs(extract_path)
#                 zip_ref.extractall(extract_path)
#                 st.write(f"文件已解压到: {os.path.abspath(extract_path)}")
#     except zipfile.BadZipFile:
#         st.error("上传的文件不是一个有效的ZIP文件。")
# else:
#     st.info("请上传一个ZIP文件")
