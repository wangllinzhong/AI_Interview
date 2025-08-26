import logging
from typing import Dict, Any

from langchain.chains.llm import LLMChain

from base.utils import load_json


class CustomLLMChain(LLMChain):
    """自定义LLMChain，允许在保存到内存前修改输出"""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # 调用父类方法获取原始输出
        result = super()._call(inputs)

        # 修改输出结果
        modified_output = self.modify_output(result[self.output_key])
        try:
            # 使用修改后的输出更新结果
            inputs['human'] = modified_output['human']
            result['ai'] = modified_output['ai']
        except Exception as e:
            logging.error(f"对话模型生成问题错误: {result}")
        return result

    def modify_output(self, output: str) -> str:
        """
        自定义修改输出的方法
        您可以在这里实现任何您需要的输出处理逻辑
        """
        # 示例：在输出前添加前缀
        try:
            json_data = load_json(output)
            return json_data
        except Exception as e:
            logging.error(f"对话模型缺少关键词导致输出错误！！！{output}")

