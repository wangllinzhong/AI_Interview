import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from langchain_core.messages import HumanMessage

from agent import AgentMasterChat
from chain import ChainMasterChat
import uuid
import shutil
from datetime import datetime
import config
import uvicorn

chat = ChainMasterChat()
app = FastAPI(title="AI面试助手", description="智能面试解决方案")
# 模拟数据库存储
interviews_db = {}
reports_db = {}

# 创建必要的目录
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.REPORT_DIR, exist_ok=True)


# 挂载静态文件
# app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页面路由，返回前端HTML"""
    with open("../frontend/index2.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api/start-interview")
async def start_interview(
        resume: UploadFile = File(None),  # 改为可选
        job_description: str = Form(""),  # 改为默认空字符串
        keywords: str = Form(""),  # 新增关键词参数
        job_title: str = Form("")  # 新增岗位名称参数
):
    """仅分析简历，不生成问题"""
    interview_id = str(uuid.uuid4())

    # 检查是否有文件上传
    file_location = None
    if resume and resume.filename:
        file_location = f"{config.UPLOAD_DIR}/{interview_id}_{resume.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(resume.file, file_object)

    interviews_db[interview_id] = {
        "file_location": file_location,
        "job_description": job_description,
        "keywords": keywords,
        "job_title": job_title,
        "status": "analyzed"
    }
    interviews = interviews_db[interview_id]
    chat.analyze_resume(interviews)
    chat.init_prompt(interviews)
    chat.init_chain()
    questions = chat.run_chain("向我提问吧！")
    # interviews_db[interview_id]["questions"] = questions['output']
    # interviews_db[interview_id]["status"] = "questions_generated"
    print(questions['text'])
    return JSONResponse({
        "success": True,
        "interview_id": interview_id,
        "first_question": questions['input']
    })


@app.post("/api/submit-answer")
async def submit_answer(request: dict):
    """提交面试问题的答案"""
    interview_id = request.get("interview_id")
    question = request.get("question")
    reply = request.get("answer")
    print(interview_id, question, reply)
    """提交面试问题的答案"""
    if interview_id not in interviews_db:
        raise HTTPException(status_code=404, detail="面试记录不存在")

    # interviews = interviews_db[interview_id]
    print(interview_id)
    questions = chat.run_chain(user_reply=reply)
    # todo 临时增加一个回复，完善逻辑后删除
    chat.memory.chat_memory.messages[-1].additional_kwargs['reply'] = reply
    # 判断问题出的是否重复
    if any(msg.content == questions.get("input", "") for msg in questions['chat_history']
           if isinstance(questions['chat_history'][0], HumanMessage)):
        questions['finished'] = True

    return JSONResponse({
        "success": True,
        "interview_id": interview_id,
        "next_question": questions['input'],
        "finished": questions['finished']
    })


@app.post("/api/finish-interview")
async def finish_interview(request: dict):
    """完成面试并生成报告"""
    try:
        # 获取面试ID
        interview_id = request.get("interview_id")

        if not interview_id:
            raise HTTPException(status_code=400, detail="缺少面试ID")

        # 验证面试记录是否存在
        if interview_id not in interviews_db:
            raise HTTPException(status_code=404, detail="面试记录不存在")

        # 检查是否已经完成
        if interviews_db[interview_id].get("status") == "completed":
            return JSONResponse({
                "success": True,
                "message": "面试已完成"
            })

        # 更新状态
        interviews_db[interview_id]["status"] = "completed"
        interviews_db[interview_id]["completed_at"] = datetime.now().isoformat()

        # 生成报告ID
        report_id = str(uuid.uuid4())
        report_path = f"{config.REPORT_DIR}/{report_id}.pdf"

        # 在实际应用中，这里会生成真实的PDF报告
        # 现在我们只是创建一个空文件作为示例
        chat_history: List[Dict[str, str]] = chat.make_pdf(report_path, report_id)

        # 存储报告信息
        reports_db[report_id] = {
            "interview_id": interview_id,
            "report_path": report_path,
            "created_at": datetime.now().isoformat(),
            "conversation_history": chat_history
        }

        # 更新面试记录中的报告ID
        interviews_db[interview_id]["report_id"] = report_id

        return JSONResponse({
            "success": True,
            "message": "面试已完成，报告生成中",
            "report_id": report_id,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/get-report")
async def get_report(request: dict):
    """获取面试报告"""
    try:
        # 从请求体中获取report_id
        report_id = request.get("new_interviewId")

        # 检查报告是否存在
        if report_id not in reports_db:
            raise HTTPException(status_code=404, detail="报告不存在")

        report_info = reports_db[report_id]

        # 返回报告数据，包含对话历史和总体评价
        return JSONResponse({
            "success": True,
            "report": {
                "conversation_history": report_info.get("conversation_history", []),
                "overall_feedback": "这是总体评价，根据实际面试表现生成"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-report/{interview_id}")
async def download_report(interview_id: str):
    """
    下载面试报告的PDF文件
    """
    try:
        # 构建PDF文件路径
        filename = f"{interview_id}.pdf"
        filepath = os.path.join(config.REPORT_DIR, filename)

        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="报告文件不存在")

        # 使用FileResponse发送文件
        return FileResponse(
            filepath,
            media_type='application/pdf',
            filename=f"AI面试报告_{interview_id}.pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        # 记录错误日志
        print(f"下载报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.get("/api/status")
async def service_status():
    """服务状态检查"""
    return JSONResponse({
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        # reload=config.DEBUG,
        # workers=config.WORKERS
    )
