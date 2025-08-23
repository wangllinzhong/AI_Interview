from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


class InterviewPromptTemplate:
    def __init__(self):
        # 处理简历提示词模板
        self.analyze_template = ""
        # 处理招聘要求提示词模板
        self.requirement_template = ""
        # 面试问答提示词模板
        self.chat_template = ""

    @property
    def analyze_prompt(self):
        analyze_template = """
        你是一名技术招聘专家，请严格按以下步骤分析简历：
        1. **技术扫描**：识别简历中提到的所有技术栈、编程语言、框架、工具、算法、方法论（如Scrum）和硬技能。
        2. **知识过滤**：排除非技术词汇（如"团队合作""本科毕业"），仅保留可通过考题验证的知识型关键词。
        3. **层级归类**：按字段归类关键词，输出格式必须是严格的JSON：
        
        {{
          "编程语言": ["Python", "Java"],
          "框架": ["Spring Boot", "React"],
          "算法": ["动态规划", "BERT"],
          "工具": ["Docker", "Kubernetes"],
          "领域知识": ["计算机视觉", "微服务架构"],
          "方法论": ["敏捷开发", "DevOps"],
          "硬技能": ["Linux基础", "SQL查询"]
        }}
        
        **简历内容**：  
        {interview}
        
        **要求**：  
        - 直接输出JSON，不添加任何额外文本
        - 关键词必须完全来自简历原文
        - 同类关键词去重（如"Python"和"python"视为重复）
        - 使用示例输出，不使用示例的关键词
        """
        return PromptTemplate(template=analyze_template, input_variables=["interview"])

    @analyze_prompt.setter
    def analyze_prompt(self, template):
        self.analyze_template = template

    @property
    def requirement_prompt(self):
        requirement_template = """
        你是一名AI面试策略引擎，请根据招聘要求生成可提问关键词：
        
        **招聘要求**：  
        {job_description}
        
        **处理规则**：  
        1. **需求解析**：从招聘要求中提取所有关键词
        2. **关键词数量**：至少30个关键词
           
        **输出格式**：
        {{
            "keywords": ["关键词1", "关键词2", "关键词3", ...]
        }}
        
        **要求**：  
        - 直接输出JSON，不添加任何额外文本
        - 同类关键词去重（如"Python"和"python"视为重复）
        - 使用示例格式输出的时候，不使用示例自带的关键词
        """
        return PromptTemplate(template=requirement_template, input_variables=["job_description", "resume_keywords_json"])

    @requirement_prompt.setter
    def requirement_prompt(self, template):
        self.requirement_template = template

    @property
    def chat_template(self):
        chat_template = """
            你是一名资深技术面试官，请基于当前关键词和历史对话生成对应的问题和答案：  
    
            **当前可提问关键词**：  
            {target_keyword}
            
            
            **提问规则**：  
            1. **问题可用**：
               - 问题必须与当前关键词相关，且能被应聘者回答
               - 问题是真实存在的面试问题或八股文，有提问的价值
            2. **深度递进**：  
               - 顺序越靠前的关键词越要重点提问
               - 遇到应聘者不会的问题，可以换一个问题
            4. **难度适配**：  
               - 高优先级关键词：设计含场景模拟的开放性问题  
               - 中低优先级：聚焦具体技术细节或对比分析  
            
            **输出规则**：  
            1. **输出格式**：输出格式必须是严格的JSON格式，可被json.loads解析，不添加任何额外文本
            2. **问题格式**：问题必须有对应的正确答案，不能凭空生成
            3. **正确答案**：正确答案尽量具体，不多于100字即可
            **输出示例**：
            {{
                "question": "您的问题内容？",
                "answer": "对应的正确答案"
            }}
        """
        return chat_template

    @chat_template.setter
    def chat_template(self, template):
        self.interview_template = template