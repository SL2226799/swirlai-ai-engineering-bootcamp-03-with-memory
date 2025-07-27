import streamlit as st
import requests

from src.chatbot_ui.core.config import config

import sys
sys.path.append("src")




st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hello! How can I assist you today?"):
    # 1. Store user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Call backend
    status, output = api_call("post", f"{config.API_URL}/rag", json={"query": prompt})
    answer = output.get("answer", "Sorry, something went wrong.")

    # 4. Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 5. Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
