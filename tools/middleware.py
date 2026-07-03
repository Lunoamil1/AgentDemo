from typing import Callable, Any

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tools(request:ToolCallRequest,handler:Callable[[ToolCallRequest],ToolMessage|Command])->ToolMessage|Command:
    logger.info(f"[tool monitor]执行的工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]执行的参数: {request.tool_call['args']}")

    try:
        result=handler(request)
        logger.info(f"[tool monitor]执行结果: {result}")
        return result
    except Exception as e:
        logger.error(f"[tool monitor]执行工具出错: {e}")
        raise e


@before_model
def log_before_model(state:AgentState,runtime:Runtime):
    msg_list = state["messages"]
    msg_count = len(msg_list)
    last_msg = msg_list[-1]

    logger.info(f"[before model]当前状态: 带有{msg_count}条消息")
    logger.debug(f"[before model]当前消息:类型为{type(last_msg).__name__} | {last_msg.content.strip()}")

    return None



# def report_prompt_swtich():
#     pass