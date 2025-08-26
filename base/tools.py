from langchain.agents import tool


@tool
def search_question():
    """返回当前的时间"""
    return ""
