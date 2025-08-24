import json
import logging
import os
import config
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.chains import SequentialChain, LLMChain, ConversationChain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from backend.utils import load_json
from prompt_template import InterviewPromptTemplate

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("AZ_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("POLO_API_BASE")


class CustomLLMChain(LLMChain):
    """自定义LLMChain，允许在保存到内存前修改输出"""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # 调用父类方法获取原始输出
        result = super()._call(inputs)

        # 修改输出结果
        modified_output = self.modify_output(result[self.output_key])
        try:
            # 使用修改后的输出更新结果
            inputs['input'] = modified_output['question']
            result[self.output_key] = modified_output['answer']
        except Exception as e:
            logging.error(f"对话模型生成问题错误: {result}")
            self.memory.chat_memory.messages[-1].additional_kwargs['state'] = True
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
            self.memory.chat_memory.messages[-1].additional_kwargs['state'] = True


class ChainMasterChat():
    """
    主聊天
    """

    def __init__(self):
        self.chat_model = ChatOpenAI(temperature=0, streaming=True, model='gpt-4o-mini-2024-07-18', max_tokens=512)
        self.model = OpenAI(temperature=0, max_tokens=512, model='gpt-3.5-turbo-instruct')
        self.template = InterviewPromptTemplate()
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
        self.memory = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key=self.MEMORY_KEY
        )
        self.analyze_chain_bad_num = 0
        self.analyze_chain_num = 0
        self.result = {'finished': False, "current_stage": "start"}

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
            HumanMessagePromptTemplate.from_template("{input}"),
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
        # 控制面试状态
        if self.result['current_stage'] == "asking":
            chain_result = self.analyze_candidate_responses(user_reply)
            self.result.update(chain_result)
        print(self.result)
        # 根据状态选择如何使用llm
        if not self.result['finished'] and self.result['current_stage'] == "start":
            chain_result = self.chain.invoke({"input": "请生成问题和答案吧！"})
            self.memory.chat_memory.messages[-2].additional_kwargs['input'] = "请生成问题和答案吧！"
            chain_result['current_stage'] = "asking"
        elif not self.result['finished'] and self.result['current_stage'] == "asking":
            chain_result = self.chain.invoke({"input": self.result['current']})
            self.memory.chat_memory.messages[-2].additional_kwargs['input'] = self.result['current']
            self.memory.chat_memory.messages[-3].additional_kwargs['reply'] = user_reply
        elif not self.result['finished'] and self.result['current_stage'] == "replying":
            if self.result['current'] == "我的提问结束了，请问你有什么想问我的吗？":
                self.memory.chat_memory.messages[-1].additional_kwargs['reply'] = user_reply
                self.result['input'] = self.result['current']
                self.result['current'] = ""
                return self.result
            chain_result= self.answer_candidate_questions(user_reply)
            self.result['input'] = chain_result['answer']
        # 如果发生错误，则停止面试
        if 'state' in self.memory.chat_memory.messages[-1].additional_kwargs:
            self.memory.chat_memory.messages.pop()
            self.memory.chat_memory.messages.pop()
            self.memory.chat_memory.messages[-1].additional_kwargs['reply'] = user_reply
            chain_result['input'] = "面试结束"
            chain_result['finished'] = True
        self.result.update(chain_result)
        print("--------------------")
        print(self.memory.buffer)
        return self.result

    def analyze_candidate_responses(self, user_reply: str = "", current_stage: str = "asking") -> dict:
        """
        1. 通过llm解析判断应聘者的回答适用于的场景（深入提问、换一个问题、结束提问、由ai回答问题、结束面试）
        2. 对应聘者的回答进行ai打分、分析应聘者的回答
        """
        load_memory = self.memory.load_memory_variables({}).get('chat_history', [])
        memory = RunnablePassthrough.assign(history=RunnableLambda(lambda x: load_memory))
        self.analyze_chain.steps[0] = memory
        result = self.analyze_chain.invoke({
            "answer": user_reply,
            "correct_answer": self.memory.chat_memory.messages[-1].content,
            "current_stage": current_stage
        })
        result_result = load_json(result)
        self.analyze_chain_num += 1
        self.analyze_chain_bad_num = self.analyze_chain_bad_num + 1 if int(result_result['ai_scoring']) < 5 else self.analyze_chain_bad_num
        if self.analyze_chain_bad_num >= 3 or self.analyze_chain_num >= 10:
            result_result.update({"current": "我的提问结束了，请问你有什么想问我的吗？", "current_stage": "replying"})
        if 'ai_scoring' in result_result:
            self.memory.chat_memory.messages[-1].additional_kwargs['ai_scoring'] = result_result['ai_scoring']
        if 'ai_comment' in result_result:
            self.memory.chat_memory.messages[-1].additional_kwargs['ai_comment'] = result_result['ai_comment']
        return result_result

    def answer_candidate_questions(self, question:str = "我没有什么问题"):
        """
        回答应聘者问题
        """
        answer_result = self.answer_chain.invoke({"question": question})
        print(answer_result)
        result = load_json(answer_result['text'])
        self.memory.chat_memory.messages[-1].content = result['answer']
        return result

    def sequential_chain_analyze_resume(self, db: dict):
        """
        使用顺序连 分析简历 -> 分析职位要求 -> 生成问题
        """
        interview = PyPDFLoader(db["file_location"])

        interview_chain = LLMChain(
            llm=self.model,
            prompt=self.template.analyze_prompt,
            output_key="resume_keywords_json",
            verbose=True
        )
        requirement_chain = LLMChain(
            llm=self.model,
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
        keywords_out = []
        interview_words_list, job_words_list, keywords_list, job_title_list = set(), set(), set(), set()
        # 对简历进行提取关键词
        if db["file_location"] is not  None:
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

    def make_pdf(self, report_path: str, interview_id: str) -> list:
        """
        读取chain的对话历史生成pdf
        """
        try:
            # 注册中文字体
            # 请确保系统中有SimHei.ttf或其他中文字体文件
            # 如果没有，可以从Windows系统复制或下载中文字体文件
            pdfmetrics.registerFont(TTFont('微软雅黑', config.TTF_FILE))
            pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))

            # 获取聊天历史
            chat_history = self.memory.buffer

            # 创建PDF文档
            doc = SimpleDocTemplate(
                report_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            # 获取样式
            styles = getSampleStyleSheet()
            # 修改标题样式
            styles['Title'].fontName = 'SimHei'
            styles['Title'].fontSize = 16
            styles['Title'].spaceAfter = 20
            styles['Title'].alignment = 1  # 居中

            # 修改一级标题样式
            styles['Heading1'].fontName = 'SimHei'
            styles['Heading1'].fontSize = 14
            styles['Heading1'].spaceAfter = 12
            styles['Heading1'].spaceBefore = 12

            styles.add(ParagraphStyle(
                name='Custom',
                parent=styles['Normal'],
                fontName='SimHei',  # 使用中文字体
                fontSize=10,
                spaceAfter=12,
                alignment=1  # 居中
            ))

            styles.add(ParagraphStyle(
                name='Question',
                parent=styles['Normal'],
                fontName='SimHei',
                fontSize=10,
                spaceAfter=6,
                leftIndent=20
            ))

            styles.add(ParagraphStyle(
                name='Answer',
                parent=styles['Normal'],
                fontName='SimHei',
                fontSize=10,
                spaceAfter=6,
                leftIndent=40
            ))

            # 构建PDF内容
            story = list()
            str_chat_history = [dict() for i in range(len(chat_history)) if i % 2 == 1]

            # 添加标题
            story.append(Paragraph("面试报告", styles['Title']))
            story.append(Spacer(1, 12))

            # 添加面试信息
            story.append(Paragraph(f"面试ID: {interview_id}", styles['Custom']))
            story.append(Paragraph(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Custom']))
            story.append(Spacer(1, 12))

            # 添加对话历史
            story.append(Paragraph("面试对话记录", styles['Heading1']))
            for i, msg in enumerate(chat_history):
                role = "面试官" if msg.type == "human" else "应聘者"
                content = msg.content

                if role == "面试官":
                    story.append(Paragraph(f"{i // 2 + 1}. 面试官：{content}", styles['Question']))
                    str_chat_history[i // 2]["question"] = content
                else:
                    story.append(Paragraph(f"应聘者：{content}", styles['Answer']))
                    str_chat_history[i // 2]["answer"] = content

                if msg.additional_kwargs.get("reply", "") == "":
                    continue
                ai = msg.additional_kwargs.get("ai_comment", "")
                str_chat_history[i // 2]["ai"] = ai
                str_chat_history[i // 2]["reply"] = msg.additional_kwargs.get("reply", "")
                story.append(Paragraph(f"AI：{ai}", styles['Answer']))
                story.append(Spacer(1, 6))

            # 构建PDF
            doc.build(story)
            print(f"PDF已生成: {report_path}")

        except Exception as e:
            print(f"生成PDF时出错: {str(e)}")
            raise
        return str_chat_history


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
