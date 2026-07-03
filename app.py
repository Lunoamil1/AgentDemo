import streamlit as st
from agent.react_agent import ReactAgent

st.title("智能扫地机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 渲染历史消息
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input():
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 流式输出助手回复
    with st.chat_message("assistant"):
        full_response = st.write_stream(st.session_state["agent"].execute_stream(prompt))

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
