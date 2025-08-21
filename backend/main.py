import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from chat import MasterChat, InterviewChat
import uuid
import shutil
from datetime import datetime
import config
import uvicorn

chat = MasterChat()
app = FastAPI(title="AI面试助手", description="智能面试解决方案")

# 创建必要的目录
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.REPORT_DIR, exist_ok=True)

# 挂载静态文件
# app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")

# 模拟数据库存储
interviews_db = {}
reports_db = {}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页面路由，返回前端HTML"""
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api/start-interview")
async def start_interview(resume: UploadFile = File(...), job_description: str = Form(...)):
    """仅分析简历，不生成问题"""
    interview_id = str(uuid.uuid4())
    file_location = f"{config.UPLOAD_DIR}/{interview_id}_{resume.filename}"

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(resume.file, file_object)

    interviews_db[interview_id] = {
        "file_location": file_location,
        "job_description": job_description,
        "status": "analyzed"
    }
    interviews = interviews_db[interview_id]
    questions = generate_sample_questions(interviews)
    # interviews_db[interview_id]["questions"] = questions['output']
    # interviews_db[interview_id]["status"] = "questions_generated"
    print(questions['output'])
    return JSONResponse({
        "success": True,
        "interview_id": interview_id,
        "first_question": questions['output']
    })


@app.post("/api/submit-answer")
async def submit_answer(request: dict):
    """提交面试问题的答案"""
    interview_id = request.get("interview_id")
    question = request.get("question")
    answer = request.get("answer")
    print(interview_id, question, answer)
    """提交面试问题的答案"""
    if interview_id not in interviews_db:
        raise HTTPException(status_code=404, detail="面试记录不存在")

    interviews = interviews_db[interview_id]
    print(interview_id)
    questions = chat.run(interviews['new_interview_keywords']['priority_keywords'])

    return JSONResponse({
        "success": True,
        "interview_id": interview_id,
        "next_question": questions['output'],
        "finished": False
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
        Path(report_path).touch()

        # 存储报告信息
        reports_db[report_id] = {
            "interview_id": interview_id,
            "report_path": report_path,
            "created_at": datetime.now().isoformat()
        }

        # 更新面试记录中的报告ID
        interviews_db[interview_id]["report_id"] = report_id

        return JSONResponse({
            "success": True,
            "message": "面试已完成，报告生成中",
            "report_id": report_id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get-report/{report_id}")
async def get_report(report_id: str):
    """获取面试报告"""
    try:
        # 检查报告是否存在
        if report_id not in reports_db:
            raise HTTPException(status_code=404, detail="报告不存在")

        report_info = reports_db[report_id]

        # 读取报告内容
        with open(report_info["report_path"], "r", encoding="utf-8") as f:
            report_content = f.read()

        return JSONResponse({
            "success": True,
            "report": {
                "questions": report_content.get("questions", []),
                "overall_feedback": report_content.get("overall_feedback", "")
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def generate_sample_questions(interviews: dict) -> list:
    """根据职位描述生成示例问题"""
    # 在实际应用中，这里会调用AI模型生成问题
    # 现在我们返回一些通用问题
    interview_chat = InterviewChat()
    interview_chat.analyze_resume(interviews)

    result = chat.run(interviews['new_interview_keywords']['priority_keywords'])
    return result


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        # reload=config.DEBUG,
        # workers=config.WORKERS
    )
