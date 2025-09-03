import hmac, sys, time
import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv


    def check_password():
        """Returns `True` if the user had a correct password."""

        def login_form():
            """Form with widgets to collect user information"""
            with st.form("Credentials"):
                st.subheader("能力中台安全测评工具")
                st.text_input("用户名", key="username")
                st.text_input("密码", type="password", key="password")
                st.form_submit_button("登录", on_click=password_entered)

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["username"] in st.secrets[
                "passwords"
            ] and hmac.compare_digest(
                st.session_state["password"],
                st.secrets.passwords[st.session_state["username"]],
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the username or password.
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        # Return True if the username + password is validated.
        if st.session_state.get("password_correct", False):
            return True

        # Show inputs for username + password.
        login_form()
        if "password_correct" in st.session_state:
            st.error("您的密码有误，请重新输入或者联系管理员")
        return False


    if not check_password():
        st.stop()

    st.set_page_config(
        "能力中台安全测评工具",
        os.path.join("img", "中台安全运营机器人方形logo.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""欢迎使用能力中台安全测评工具！"""
        }
    )

    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "能力中台安全测评工具.png"
            ),
            use_container_width=True
        )
        # st.caption(
        #     f"""测试版V2，仅供内部使用""",
        #     unsafe_allow_html=True,
        # )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
