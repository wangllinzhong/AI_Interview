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
        # 应聘者回答分析模板
        self.answer_template = ""
        # 面试官回复模板
        self.interviewer_template = ""
        # 通用模板
        self.general_template = ""

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
            你是一名AI面试策略引擎，你会听从对话中的指令，根据指令对关键词生成相关的问题和答案：  
    
            **关键词**：  
            {target_keyword}
            
            
            **提问规则**：  
            1. 输出结果中必须包含问题和对应的答案，一个都不能少
            2. 同一个关键词的问题在不深入提问的情况下，生成基础问题和答案
            3. 指令要求深入提问，你就要使用上一个关键词深入生成该关键词对应的问题和答案
            4. 指令要求换一个问题，你就要使用下一个关键词生成该关键词对应的问题和答案
            5. 如果指令要求换一个问题，但是后续没有可提问的关键词时，可以在所有关键词中随机选择生成对应的问题和答案
            6. 所有提问的问题和答案均不允许与历史记录重复
            7. 按照关键词的先后顺序作为优先级，优先级越高越先提问
            8. 不需要回复对话，输出结果只能是与关键词相关的问题和答案
            9. 答案的字数控制在100字以内
            10. 输出格式必须严格遵守JSON格式，必要文本不得增多或缺失

            
            **输出格式**：
            ```
            {{
                "question": "您的问题内容？",
                "answer": "对应的正确答案"
            }}
            ```
        """
        return chat_template

    @chat_template.setter
    def chat_template(self, template):
        self._chat_template = template


    @property
    def answer_template(self):
        # [深入提问、换一个问题、结束提问、由ai回答问题、结束面试]
        template = """
            你是一名资深技术面试官，面试分为两个环节技术面试和应聘者提问，你需要根据当前的面试环节判断需要做的任务：
            **评分与评语**：
            1. 根据**应聘者回答**和**正确答案**的匹配度、回答正确性、综合评价三方面对应聘者回答的问题进行评分和评语
            2. 评分处于0-10之间，评分越高代表应聘者回答越接近答案。
            3. 评语需要简短，字数控制在50字以内，评语需要包含应聘者回答的优缺点，以及应聘者回答的不足之处。
            **技术面试**：
            1. 如果**应聘者回答**与**正确答案**匹配度较高、回答正确性较高或综合评价较高，你需要返回JSON格式：
                {{
                    "current_stage": "asking",
                    "current": "请继续深入提问",
                    "ai_scoring": "评分",
                    "ai_comment": "评语"
                }}
            2. 如果**应聘者回答**与**正确答案**匹配度较低、回答正确性较差或综合评价一般，你需要返回JSON格式：
                {{
                    "current_stage": "asking",
                    "current": "换一个问题继续提问",
                    "ai_scoring": "评分",
                    "ai_comment": "评语"
                }}
            3. 如果**历史记录**中的对话超过10对，你需要返回JSON格式：
                {{
                    "current_stage": "replying",
                    "current": "我的提问结束了，请问你有什么想问我的吗？",
                    "ai_scoring": "评分",
                    "ai_comment": "评语"
                }}
            4. 如果**应聘者回答**有至少三道回答正确性较差，你需要返回JSON格式：
                {{
                    "current_stage": "replying",
                    "current": "我的提问结束了，请问你有什么想问我的吗？",
                    "ai_scoring": "评分",
                    "ai_comment": "评语"
                }}
            
            **历史记录**：
            {history}

            **应聘者回答**：  
            {answer}

            **正确答案**：
            {correct_answer}
            
            **面试环节**
            {current_stage}

            **评价规则**：  
            1. **回答质量**：回答是否准确、全面、深入，是否体现了应聘者的技术深度和广度
            2. **逻辑清晰**：回答是否条理清晰，逻辑性强，是否能够自洽
            3. **问题匹配**：回答是否与提问的关键词相关，是否能够回答出问题的本质
            4. **回答正确性**： 与正确答案是否相似
            4. **综合评价**：给出综合评价
        """
        return PromptTemplate(template=template, input_variables=["answer", "correct_answer", "question_num", "current_stage"])


    @answer_template.setter
    def answer_template(self, template):
        self._answer_template = template

    @property
    def interview_template(self):
        template = """
            你是一名资深技术面试官，你可以以成都当地的互联网科技公司的标准水平对应聘者提出的问题作出答复：  
            **应聘者问题**：  
            {question}
            
            **输出规则**：  
            1. **输出格式**：输出格式必须是严格的JSON格式，可被json.loads解析，不添加任何额外文本
            2. 如果应聘者表示没有问题了、不想提问了则finished字段改为True
            **输出示例**：
            {{
                "question": "应聘者问题内容？",
                "answer": "你的回答",
                "finished": false
            }}
       """
        return PromptTemplate(template=template, input_variables=["question"])

    @interview_template.setter
    def interview_template(self, template):
        self._interview_template = template

    @property
    def general_template(self):
        general_template = """
            你是一个面试官，你可以根据招聘岗位生成至少15个适合该岗位的技术关键词。
            
            **招聘岗位**：
            {job_title}
            
            **输出**：  
            {{
                "keywords": ["关键词1", "关键词2", "关键词3", ...]
            }}
        """
        return PromptTemplate(template=general_template, input_variables=["job_title"])

    @general_template.setter
    def general_template(self, template):
        self._general_template = template