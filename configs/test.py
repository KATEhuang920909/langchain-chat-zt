import streamlit as st

# 注入自定义 CSS，优化文件上传组件的样式
css = '''
<style>
    /* 拖放区域提示文本 */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "将文件拖放到此处";
    }

    /* 隐藏默认的提示文本 */
    [data-testid="stFileUploaderDropzone"] div div span {
        display: none !important;
    }

    /* 文件大小及类型提示 */
    [data-testid="stFileUploaderDropzone"] div div::after {
        color: rgba(49, 51, 63, 0.6);
        font-size: .8em;
        content: "每个文件限制200MB•XLSX";
    }

    /* 隐藏默认的文件大小提示 */
    [data-testid="stFileUploaderDropzone"] div div small {
        display: none !important;
    }

    /* 处理按钮文本 - 尝试不同的选择器组合 */
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }

    [data-testid="stFileUploaderDropzone"] button::after {
        content: "浏览文件";
        font-size: 17px !important;
    }

    /* 额外选择器，确保覆盖所有可能的按钮文本 */
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton"] {
        font-size: 0 !important;
    }

    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton"]::after {
        content: "浏览文件";
        font-size: 17px !important;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)


# 创建文件上传组件
uploaded_file = st.file_uploader(
    "请上传ZIP压缩包",
    type="zip",
    accept_multiple_files=False,
    help="格式：html,htm,mhtml,md,json,jsonl,csv,pdf,docx,txt,ppt,pptx,png,jpg,jpeg,bmp,eml,msg,rst,rtf,xml,epub,odt,tsv,eml,msg,epub,xlsx,xls,xlsd,ipynb,odt,py,rst,rtf,srt,toml,tsv,xml,ppt,pptx,enex"
)

# 测试：显示上传结果
if uploaded_file is not None:
    st.success(f"已上传文件：{uploaded_file.name}")