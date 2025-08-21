# 项目配置需求
1. miniconda
2. python 3.11.13
3. conda install -r requirements.txt
4. langchain_openai 无法用conda安装需要用pip子项安装
5. dotenv 无法用conda安装需要用pip子项安装

# agent流程
1. 用户输入简历（pdf）和面试岗位需求
2. 服务器接口调用langchain，将简历和面试岗位需求作为输入，返回面试官提问的问题和答案
    1. 解析面试岗位需求
    2. 解析简历
    3. 生成面试官提问的问题和答案（动态问题生成）
3. 客户端接收问题，展示给面试人员提问
4. 面试人员回答问题，客户端接收答案
5. langchain通过Rule-based + LLM-based 决策机制，判断进一步提问、另起一个问题或提问结束
    1. Rule-based：基于规则进行硬性条件过滤
    2. LLM-based：基于语言模型进行软性条件过滤
6. 面试结束后整理面试中的面试官问题、面试者回答和正确答案，生成面试报告（pdf）

![alt text](deepseek_mermaid_20250818_97c866.png)


# 服务器接口 -> langchain -> deepseek
# 客户端：面试官
# 接口：http、https、websocket

# 服务器
1. 接口访问，python选择fastapi