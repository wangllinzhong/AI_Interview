import json
import logging
import os

from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor, initialize_agent, AgentType
from langchain.chains import SequentialChain, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StdOutCallbackHandler, FileCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool
from langchain_openai import OpenAI, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from tools import search_question
from prompt_template import InterviewPromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("AZ_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("POLO_API_BASE")


# os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
# os.environ["OPENAI_API_BASE"] = os.getenv("DEEPSEEK_API_BASE")

# 设置文件日志
# logging.basicConfig(level=logging.INFO)
# file_handler = FileCallbackHandler("../log/lcel_log.log")

class AgentMasterChat():
    """
    主聊天
    """

    def __init__(self):
        # 初始化MasterChat类
        # todo 目前使用openai 后续可以根据需求更改
        self.chat_model = ChatOpenAI(temperature=0, streaming=True)
        self.template = InterviewPromptTemplate()
        # 设置聊天历史记录的键名
        self.MEMORY_KEY = "chat_history"
        # 工具列表初始化为空
        self.tools = [Tool(
            name="search_question",
            func=search_question,
            description="返回当前的时间"
        )]

        self.prompt = None
        self.agent_executor = None

    def run(self, input: str):
        """
        运行聊天
        """
        result = self.agent_executor.invoke({"input": input})
        return result

    def init_agent(self):
        """
        初始化agent启动器
        """
        memory = ConversationBufferMemory(
            # llm=self.chat_model,
            human_prefix="user",
            ai_prefix="ai",
            memory_key=self.MEMORY_KEY,
            return_messages=True,
            output_key='output',
            max_token_limit=1000
        )
        agent = create_openai_tools_agent(
            tools=self.tools,
            llm=self.chat_model,
            prompt=self.prompt,
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            max_interactions=1,  # 最大迭代次数为1，防止反复调用
            early_stopping_method="force",  # 强制停止
        )

    def init_prompt(self, keywords: dict):
        """
        初始化提示词prompt
        """
        system_chat_template = self.template.chat_template.format(target_keyword=keywords['new_interview_keywords'])
        self.prompt = ChatPromptTemplate([
            SystemMessage(system_chat_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        # result = self.prompt | self.chat_model | StrOutputParser()
        # result.invoke({input: "请根据我可提问的关键词，帮我生成面试问题向我提出一个问题吧！"})

    def chain_analyze_resume(self, db: dict):
        """
        使用顺序连 分析简历 -> 分析职位要求 -> 生成问题
        """
        interview = PyPDFLoader(db["file_location"])
        model = OpenAI(temperature=0, max_tokens=512)

        interview_chain = LLMChain(
            llm=model,
            prompt=self.template.analyze_prompt,
            output_key="resume_keywords_json",
            verbose=True
        )
        requirement_chain = LLMChain(
            llm=model,
            prompt=self.template.requirement_prompt,
            output_key="priority_keywords",
            verbose=True
        )

        sequential_chain = SequentialChain(
            chains=[interview_chain, requirement_chain],
            input_variables=["interview", "job_description"],
            output_variables=["resume_keywords_json", "priority_keywords"],
            verbose=True,
        )

        result = sequential_chain.invoke({
            "interview": interview.load()[0].page_content,
            "job_description": db["job_description"],
        })
        db['new_interview_keywords'] = result

    def analyze_resume(self, db: dict):
        """
        使用顺序连 分析简历 -> 生成问题
        """
        interview = PyPDFLoader(db["file_location"])
        model = OpenAI(temperature=0, max_tokens=512)

        chain = self.template.analyze_prompt | model

        # 使用callbacks记录日志
        # result = chain.invoke({"interview": interview.load()[0].page_content},
        #                       config={"callbacks":[StdOutCallbackHandler(), file_handler]})
        result = chain.invoke({"interview": interview.load()[0].page_content})
        db['new_interview_keywords'] = result


if __name__ == "__main__":
    job_requirement = """
    「岗位职责」\r\n1.基于LangChain进行大模型应用开发，构建智能问答、知识管理、自动化流程等AI解决方案\r\n2.负责Agent系统的设计与开发，包括任务规划、多轮对话管理、意图识别等核心模块实现\r\n3.构建企业级RAG系统，实现非结构化文档解析、向量化检索（FAISS/Chroma）与生成优化\r\n4.基于Coze等平台开发工作流，集成大模型能力实现业务流程自动化，设计低延迟推理服务与异步处理机制\r\n5.进行模型微调与部署优化，结合业务场景平衡性能与推理成本\r\n6.跟踪Transformer/GPT等架构演进，探索Agent+RAG+工作流的技术融合创新\r\n「任职要求」\r\n1.计算机/人工智能相关专业本科及以上学历，2年以上AI应用开发经验\r\n2.精通Python，熟悉PyTorch，具备扎实的数据结构与算法基础\r\n3.熟练掌握LangChain等开发框架，具备RAG系统从0到1搭建经验\r\n4.熟悉Agent开发全流程，有Dify/Coze等平台，熟悉LangGraph并有实战经验\r\n5.具备工作流平台开发经验，熟悉Elasticsearch/Milvus等检索技术及Kafka等消息中间件\r\n6.熟悉云计算部署，掌握Docker以及Kubernetes\r\n7.优秀的工程化能力，能完成模型量化（GGML）、推理加速（vLLM）等性能优化
    """
    file_location = r"D:\\code\\python\\AI-Interview\\backend\\static/uploads/82504130-96ae-4721-8f64-221845b5cb06_简历-王林重-ai.pdf"
    interviews_db: dict = {"file_location": file_location, "job_description": job_requirement}

    interview_chat = MasterChat()
    interview_chat.analyze_resume(interviews_db)

    chat = MasterChat()
    result = chat.run(interviews_db['new_interview_keywords']['priority_keywords'])
    print(result)
