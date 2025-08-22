import json
import os
import config
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.chains import SequentialChain, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain.memory import ConversationBufferMemory
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

        # 使用修改后的输出更新结果
        inputs['input'] = modified_output['question']
        result[self.output_key] = modified_output['answer']

        return result

    def modify_output(self, output: str) -> str:
        """
        自定义修改输出的方法
        您可以在这里实现任何您需要的输出处理逻辑
        """
        # 示例：在输出前添加前缀
        return json.loads(output)


class ChainMasterChat():
    """
    主聊天
    """

    def __init__(self):
        self.chat_model = ChatOpenAI(temperature=0, streaming=True)
        self.template = InterviewPromptTemplate()
        self.MEMORY_KEY = "chat_history"
        # 移除tools，因为我们不需要工具调用
        self.tools = []
        self.prompt = None
        self.chain = None  # 改为chain
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key=self.MEMORY_KEY
        )
        self.num = 0

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
        # 创建链
        self.chain = CustomLLMChain(
            llm=self.chat_model,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )

    def run_chain(self, input: str = "请继续提问！", user_reply: str = ""):
        """
        运行聊天
        """
        result = self.chain.invoke({"input": input})
        self.memory.chat_memory.messages[-2].additional_kwargs['input'] = input
        if user_reply:
            self.memory.chat_memory.messages[-1].additional_kwargs['reply'] = user_reply
        result['finished'] = False
        print("--------------------")
        print(self.memory.buffer)
        return result

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

    def make_pdf(self, report_path: str, interview_id: str):
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

            # 添加标题
            story.append(Paragraph("面试报告", styles['Title']))
            story.append(Spacer(1, 12))

            # 添加面试信息
            story.append(Paragraph(f"面试ID: {interview_id}", styles['Custom']))
            story.append(Paragraph(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Custom']))
            story.append(Spacer(1, 12))

            # 添加对话历史
            story.append(Paragraph("面试对话记录", styles['Heading1']))
            for i, msg in enumerate(chat_history, 1):
                role = "面试官" if msg.type == "human" else "应聘者"
                content = msg.content

                if role == "面试官":
                    story.append(Paragraph(f"{i}. 面试官：{content}", styles['Question']))
                else:
                    story.append(Paragraph(f"{i}. 应聘者：{content}", styles['Answer']))
                story.append(Spacer(1, 6))

            # 构建PDF
            doc.build(story)
            print(f"PDF已生成: {report_path}")

        except Exception as e:
            print(f"生成PDF时出错: {str(e)}")
            raise
        return story


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
