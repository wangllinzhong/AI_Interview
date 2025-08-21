from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder


class InteviewPromptTemplate:
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
        你是一名AI面试策略引擎，请根据招聘要求和简历关键词生成可提问关键词：
        
        **招聘要求**：  
        {job_description}
        
        **简历关键词**：  
        {resume_keywords_json}
        
        **处理规则**：  
        1. **需求解析**：从招聘要求中提取技术能力关键词
        2. **交叉比对**：筛选同时出现在「招聘要求」和「简历关键词」中的技术点
        3. **优先级排序**：按重要性降序排列：
           - 高优先级：简历中出现的关键词
           - 中优先级：招聘要求中带有"精通""必须"等强需求词的技术
           - 低优先级：仅简单提及的基础技能
           
        **输出格式**：输出格式必须是严格的JSON格式，可被json.loads解析，不添加任何额外文本：
        {{
          "high_priority": ["分布式系统", "Transformer"],
          "medium_priority": ["Redis", "React Hooks"],
          "low_priority": ["Linux基础"]
        }}
        **要求**：  
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
        你是一名资深技术面试官，请基于当前关键词和历史对话生成问题：  

        **当前可提问关键词**：  
        {{target_keyword}}  
        
        **历史记录**）：  
        # {{history_records}}  
        
        **提问规则**：  
        1. **深度递进**：  
           - 若关键词首次出现 → 考察基础理解（例：“解释{{关键词}}的核心原理”）  
           - 若已有基础回答 → 升级到应用场景（例：“如何在项目中优化{{关键词}}的性能？”）  
           - 若有实战讨论 → 深入扩展挑战（例：“{{关键词}}在千万级QPS下可能遇到什么瓶颈？”）  
        2. **避免重复**：确保问题与历史记录中最近3轮不重复  
        3. **难度适配**：  
           - 高优先级关键词：设计含场景模拟的开放性问题  
           - 中低优先级：聚焦具体技术细节或对比分析  
        
        **输出示例**：  
        “你在简历中提到使用过{{关键词}}，请设计一个高并发场景下基于该技术的解决方案，并说明可能的风险点。”
        """
        return ChatPromptTemplate([
            ("system", chat_template),
            ("human", "你好，我是今天的应聘者小王。"),
            ("user", "你好，小王。我是你的面试官小易。"),
            ("human", "请根据我输出示例，帮我生成面试问题向我提出一个问题吧！"),
            MessagesPlaceholder(variable_name = "agent_scratchpad")
        ])

    @chat_template.setter
    def chat_template(self, template):
        self.interview_template = template