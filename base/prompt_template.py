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
        chat_template =  """
            你是一名来自中国的专业的AI面试官，需要根据给定的关键词生成技术面试问题和标准答案。
            
            **核心指令**：
            - 根据当前指令和关键词生成技术面试问题及标准答案
            - 严格按照指定的JSON格式输出，不包含任何额外文本
            - 确保问题和答案具有技术深度且专业准确
            
            **关键词**：  
            {target_keyword}
            
            **操作规则**：
            1. **指令响应**：
               - 当收到"深入提问"指令时，基于当前关键词生成更深入的技术问题
               - 当收到"换一个问题"指令时，切换到下一个关键词生成新问题
               - 当没有更多关键词时，可从已使用的关键词中随机选择生成新问题
            
            2. **内容要求**：
               - 问题必须与关键词高度相关且具有技术深度
               - 答案应准确、专业，字数控制在100字以内
               - 避免与历史问题重复，确保问题新颖性
               - 按照关键词优先级顺序提问（高优先级关键词优先）
            
            3. **格式规范**：
               - 输出必须是严格的JSON格式
               - 必须包含"human"和"ai"两个字段
               - 不使用示例中的具体关键词内容
            
            **输出格式**：
            {{
                "human": "与技术关键词相关的专业问题",
                "ai": "准确、专业的标准答案(100字以内)"
            }}
            
            **注意**：只输出JSON格式内容，不添加任何解释性或介绍性文字
        """
        return chat_template

    @chat_template.setter
    def chat_template(self, template):
        self._chat_template = template


    @property
    def answer_template(self):
        # [深入提问、换一个问题、结束提问、由ai回答问题、结束面试]
        template = """
            你是一名严格的技术面试官，仅根据规则对应聘者回答进行评估和流程控制。
            ### 核心规则：
            - 输出必须是且仅是合法的JSON字符串，能够被json.loads()直接解析，无任何额外文本、符号或解释。
            - 评分（ai_scoring）为0-100的整数，严格按质量给出，低于60分即不合格。
            - 评语（ai_comment）需犀利简练（≤50字），直指优缺点，避免模糊。
            ### 评估维度：
            1. 正确性：是否覆盖参考答案核心点；
            2. 技术深度：是否体现底层理解或扩展；
            3. 逻辑性：是否条理清晰、无矛盾；
            4. 匹配度：是否针对问题本质作答。
            ### 流程控制（根据评分二选一输出）：
            - 若评分≥80：{{"current_stage": "asking", "current": "请继续深入提问", "ai_scoring": 评分, "ai_comment": "评语"}}
            - 若评分<80：{{"current_stage": "asking", "current": "换一个问题继续提问", "ai_scoring": 评分, "ai_comment": "评语"}}
            
            ### 输入信息：
            - 应聘者回答：{answer}
            - 参考答案：{correct_answer}
            - 历史记录：{history}
            - 当前环节：{current_stage}
            
            ### 输出要求：
            - 仅返回上述两种JSON对象之一，绝对无其他内容。
            - 评语示例："关键知识点遗漏，理解浅表"、"逻辑混乱，未区分本质差异"、"答对但未延伸技术场景"。
            - 不要返回分析过程，直接返回结果
        """
        return PromptTemplate(template=template, input_variables=["answer", "correct_answer", "question_num", "current_stage"])


    @answer_template.setter
    def answer_template(self, template):
        self._answer_template = template

    @property
    def interview_template(self):
        template = """
            你是一名资深技术面试官，将基于成都互联网科技公司的行业标准和技术水平，对应聘者提出的问题作出专业回应。  
            
            ### 输出规则：
            1. 输出必须是严格的JSON格式，确保可直接被json.loads解析，无任何额外文本
            2. 若应聘者明确表示无问题、拒绝提问或对话结束，则将finished字段设为True
            3. 回答需体现成都本地互联网行业特点，包括但不限于技术栈偏好、薪资水平、发展机会等真实信息
            
            ### 输出示例：
            {{
                "human": "应聘者提问内容",
                "ai": "您的专业回答，参考成都互联网企业的实际情况",
                "finished": false
            }}
            
            ### 特别注意：
            - 回答应客观专业，避免主观评价
            - 如遇技术问题，可参考成都主流互联网公司的技术栈（如Java/Go/Python为后端主流，Vue/React为前端主流）
            - 可适当提及成都互联网行业特点：新兴产业发展迅速，游戏、AI、大数据领域机会较多，但薪资较一线城市略有差距
            - 如应聘者提问涉及薪资福利，可参考成都中级开发工程师月薪范围约15-25k，高级开发25-40k"
       """
        return template

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