from datetime import datetime
from typing import Any

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationBufferMemory
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import config


class EnhanceConversationMemory(ConversationBufferMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._full_history: list = list()

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        super().save_context(inputs, outputs)

        # 同时保存到完整历史
        human_input = list(inputs.values())[0] if inputs else ""
        ai_output = list(outputs.values())[1] if outputs else ""

        self._full_history.append({"human_input": human_input, "ai_output": ai_output})
        del self.chat_memory.messages[-1]

    def save_history(self, path: str, interview_id=None):
        """
        把历史记录存入pdf中
        """
        try:
            # 注册中文字体
            # 请确保系统中有SimHei.ttf或其他中文字体文件
            # 如果没有，可以从Windows系统复制或下载中文字体文件
            pdfmetrics.registerFont(TTFont('微软雅黑', config.TTF_FILE))
            pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))

            # 获取聊天历史
            chat_history = self.memory.full_history

            # 创建PDF文档
            doc = SimpleDocTemplate(
                path,
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

            # 添加标题
            story.append(Paragraph("面试报告", styles['Title']))
            story.append(Spacer(1, 12))

            # 添加面试信息
            story.append(Paragraph(f"面试ID: {interview_id}", styles['Custom']))
            story.append(Paragraph(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Custom']))
            story.append(Spacer(1, 12))

            # 添加对话历史
            story.append(Paragraph("面试对话记录", styles['Heading1']))
            for i, msg in enumerate(self.full_history):
                if msg['state'] in ['start', 'asking']:
                    story.append(Paragraph(f"{i + 1}. 面试官：{msg['input']}", styles['Question']))
                    story.append(Paragraph(f"应聘者：{msg['output']}", styles['Answer']))
                    story.append(Paragraph(f"AI：{msg['ai']}", styles['Answer']))
                else:
                    story.append(Paragraph(f"{i + 1}. 面试官：{msg['input']}", styles['Question']))
                    story.append(Paragraph(f"应聘者：{msg['output']}", styles['Answer']))
                story.append(Spacer(1, 6))

            # 构建PDF
            doc.build(story)
            print(f"PDF已生成: {path}")

        except Exception as e:
            print(f"生成PDF时出错: {str(e)}")
            raise

    @property
    def full_history(self):
        return self._full_history

    @full_history.setter
    def full_history(self, full_history):
        self._full_history = full_history
