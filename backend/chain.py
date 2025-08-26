import json
import os
import config
from datetime import datetime

from dotenv import load_dotenv
from langchain.chains import SequentialChain, LLMChain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import OpenAI, ChatOpenAI

from base.struct_chain import CustomLLMChain
from base.struct_callback import HistoryCallback
from base.struct_memory import EnhanceConversationMemory
from base.utils import load_json
from base.prompt_template import InterviewPromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("AZ_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("POLO_API_BASE")


class ChainMasterChat:
    """
    主聊天
    """

    def __init__(self):
        self.chat_model = ChatOpenAI(temperature=0, streaming=True, model='gpt-4o-mini-2024-07-18', max_tokens=512)
        self.model = OpenAI(temperature=0, max_tokens=512, model='gpt-3.5-turbo-instruct')
        self.template = InterviewPromptTemplate()
        self.callbacks = [HistoryCallback()]
        self.MEMORY_KEY = "chat_history"
        # 移除tools，因为我们不需要工具调用
        self.tools = []
        self.prompt = None
        self.chain = None  # 改为chain
        self.analyze_chain = None
        self.answer_chain = None
        # self.memory = ConversationBufferMemory(
        #     return_messages=True,
        #     memory_key=self.MEMORY_KEY
        # )
        self.memory = EnhanceConversationMemory(
            llm=self.chat_model,
            return_messages=True,
            memory_key=self.MEMORY_KEY,
            output_key="ai",
            verbose=True
        )
        self.analyze_chain_bad_num = 0
        self.analyze_chain_num = 0
        self.chain_result = {"finished": False, 'current_stage':  'start'}

    def init_prompt(self, keywords: dict):
        """
        初始化提示词prompt
        """
        system_chat_template = self.template.chat_template.format(
            target_keyword=json.dumps(keywords['new_interview_keywords'], ensure_ascii=False)
        )

        # 构建正确的ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_chat_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human}"),
        ])

    def init_chain(self):
        """
        初始化chain而不是agent
        """
        # 创建简单的chain
        # self.chain = self.prompt | self.chat_model | StrOutputParser()
        # 用于对话
        self.chain = CustomLLMChain(
            llm=self.chat_model,
            prompt=self.prompt,
            memory=self.memory,
            # callbacks=self.callbacks,
            verbose=True
        )
        # 用于分析应聘者的回答情况
        load_memory = self.memory.load_memory_variables({}).get('chat_history', [])
        memory = RunnablePassthrough.assign(history=RunnableLambda(lambda x: load_memory))
        self.analyze_chain = memory | self.template.answer_template | self.model
        # 用于回答应聘者问题
        # self.answer_chain = self.template.interview_template | self.chat_model | StrOutputParser
        self.answer_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.template.interview_template,
            memory=self.memory,
            verbose=True
        )

    def run_chain(self, user_reply: str = "") -> dict:
        """
        运行聊天
        """
        self.chain_result['current'] = user_reply if user_reply != "" else "请生成问题和答案吧！"
        # 控制面试状态
        if self.chain_result['current_stage'] == "asking":
            self.memory.chat_memory.messages.append(AIMessage(content=self.chain_result['current']))
            self.memory.full_history[-1].update({'reply': user_reply, 'current': self.chain_result['current']})
            # 评估应聘者回答
            self.chain_result.update(self.analyze_candidate_responses())
        print(self.chain_result)
        # 根据状态选择如何使用llm
        if not self.chain_result['finished'] and self.chain_result['current_stage'] == "start":
            self.chain_result.update(self.chain.invoke({"human": self.chain_result['current']}))
            self.chain_result['current_stage'] = "asking"
        elif not self.chain_result['finished'] and self.chain_result['current_stage'] == "asking":
            self.chain_result.update(self.chain.invoke({"human": self.chain_result['current']}))
        elif not self.chain_result['finished'] and self.chain_result['current_stage'] == "replying":
            if self.chain_result['current'] == "我的提问结束了，请问你有什么想问我的吗？":
                self.chain_result['human'] = self.chain_result['current']
                return self.chain_result
            self.chain_result = self.answer_candidate_questions(self.chain_result['reply'])
        else:
            self.chain_result['ai'] = "面试结束"
            self.chain_result['finished'] = True
        print("--------------------")
        print(self.memory.full_history)
        print(self.memory.buffer)
        return self.chain_result

    def analyze_candidate_responses(self) -> dict:
        """
        1. 通过llm解析判断应聘者的回答适用于的场景（深入提问、换一个问题、结束提问、由ai回答问题、结束面试）
        2. 对应聘者的回答进行ai打分、分析应聘者的回答
        """
        memory = RunnablePassthrough.assign(history=RunnableLambda(lambda x: self.memory.buffer))
        self.analyze_chain.steps[0] = memory
        result = self.analyze_chain.invoke({
            "answer": self.memory.full_history[-1]['reply'],
            "correct_answer": self.memory.full_history[-1]['ai_output'],
            "current_stage": self.chain_result['current_stage']
        })
        result_result = load_json(result)
        self.analyze_chain_num += 1
        self.analyze_chain_bad_num = self.analyze_chain_bad_num + 1 if int(
            result_result['ai_scoring']) < 55 else self.analyze_chain_bad_num
        if self.analyze_chain_bad_num >= 3 or self.analyze_chain_num >= 10:
            result_result.update({"current": "我的提问结束了，请问你有什么想问我的吗？", "current_stage": "replying"})
        if 'ai_scoring' in result_result:
            self.memory.chat_memory.messages[-1].additional_kwargs['ai_scoring'] = result_result['ai_scoring']
        if 'ai_comment' in result_result:
            self.memory.chat_memory.messages[-1].additional_kwargs['ai_comment'] = result_result['ai_comment']
        return result_result

    def answer_candidate_questions(self, question: str = "我没有什么问题"):
        """
        回答应聘者问题
        """
        answer_result = self.answer_chain.invoke({"question": question})
        print(answer_result)
        result = load_json(answer_result['text'])
        return result

    def analyze_resume(self, db: dict):
        """
        使用顺序连 分析简历 -> 生成问题
        """
        keywords_out = []
        interview_words_list, job_words_list, keywords_list, job_title_list = set(), set(), set(), set()
        # 对简历进行提取关键词
        if db["file_location"] is not None:
            interview = PyPDFLoader(db["file_location"])
            interview_chain = self.template.analyze_prompt | self.model
            interview_words = interview_chain.invoke({"interview": interview.load()[0].page_content})
            words_json = load_json(interview_words)
            interview_words_list = set([o for i in words_json.values() for o in i])

        if db['job_description'] != "":
            description_chain = self.template.requirement_prompt | self.model
            job_words = description_chain.invoke({"job_description": db['job_description']})
            words_json = load_json(job_words)
            job_words_list = set([o for i in words_json.values() for o in i])

        if db['keywords'] != "":
            keywords_list = set(db['keywords'].split(",") if "," in db['keywords'] else db['keywords'].split("，"))

        if db['job_title'] != "":
            job_chain = self.template.general_template | self.model
            job_words = job_chain.invoke({"job_title": db['job_title']})
            words_json = load_json(job_words)
            job_title_list = set([o for i in words_json.values() for o in i])

        keywords_out.extend(keywords_list)
        keywords_out.extend(job_words_list & interview_words_list)
        keywords_out.extend(interview_words_list - job_words_list)
        keywords_out.extend(job_title_list)
        keywords_out.extend(job_words_list - interview_words_list)

        db['new_interview_keywords'] = keywords_out


if __name__ == "__main__":
    job_requirement = """
    「岗位职责」\r\n1.基于LangChain进行大模型应用开发，构建智能问答、知识管理、自动化流程等AI解决方案\r\n2.负责Agent系统的设计与开发，包括任务规划、多轮对话管理、意图识别等核心模块实现\r\n3.构建企业级RAG系统，实现非结构化文档解析、向量化检索（FAISS/Chroma）与生成优化\r\n4.基于Coze等平台开发工作流，集成大模型能力实现业务流程自动化，设计低延迟推理服务与异步处理机制\r\n5.进行模型微调与部署优化，结合业务场景平衡性能与推理成本\r\n6.跟踪Transformer/GPT等架构演进，探索Agent+RAG+工作流的技术融合创新\r\n「任职要求」\r\n1.计算机/人工智能相关专业本科及以上学历，2年以上AI应用开发经验\r\n2.精通Python，熟悉PyTorch，具备扎实的数据结构与算法基础\r\n3.熟练掌握LangChain等开发框架，具备RAG系统从0到1搭建经验\r\n4.熟悉Agent开发全流程，有Dify/Coze等平台，熟悉LangGraph并有实战经验\r\n5.具备工作流平台开发经验，熟悉Elasticsearch/Milvus等检索技术及Kafka等消息中间件\r\n6.熟悉云计算部署，掌握Docker以及Kubernetes\r\n7.优秀的工程化能力，能完成模型量化（GGML）、推理加速（vLLM）等性能优化
    """
    file_location = r"D:\\code\\python\\AI-Interview\\backend\\static/uploads/82504130-96ae-4721-8f64-221845b5cb06_简历-王林重-ai.pdf"
    interviews_db: dict = {"file_location": file_location, "job_description": job_requirement}

    interview_chat = ChainMasterChat()
    interview_chat.analyze_resume(interviews_db)

    chat = ChainMasterChat()
    result = chat.run(interviews_db['new_interview_keywords']['priority_keywords'])
    print(result)
