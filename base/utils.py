import json
import logging


def load_json(jsons: str):
    """
    将llm返回的字符串类型的json格式数据变为字典返回
    """
    try:
        s_index = jsons.find("{")
        e_index = jsons.rfind("}") + 1
        new_json = jsons[s_index:e_index]
        return json.loads(new_json)
    except Exception as e:
        logging.error(f"解析输出json数据错误：{jsons}")