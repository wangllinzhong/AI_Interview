import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from typing import Optional
import uuid
from pathlib import Path
import shutil
import json
from datetime import datetime
import config

app = FastAPI(title="AI面试助手", description="智能面试解决方案")

# 创建必要的目录
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.REPORT_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# 模拟数据库存储
interviews_db = {}
reports_db = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页面路由，返回前端HTML"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """分析简历和职位描述"""
    try:
        # 生成唯一ID
        interview_id = str(uuid.uuid4())
        
        # 保存上传的文件
        file_location = f"{config.UPLOAD_DIR}/{interview_id}_{resume.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(resume.file, file_object)
        
        # 存储面试信息
        interviews_db[interview_id] = {
            "resume_path": file_location,
            "job_description": job_description,
            "created_at": datetime.now().isoformat(),
            "status": "analyzed",
            "questions": []
        }
        
        # 模拟分析过程 - 实际应用中这里会调用AI模型
        questions = generate_sample_questions(job_description)
        interviews_db[interview_id]["questions"] = questions
        
        print(interviews_db[interview_id]) 

        return JSONResponse({
            "success": True,
            "interview_id": interview_id,
            "questions": questions,
            "message": "简历分析完成"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/submit-answer")
# async def submit_answer(
#     interview_id: str = Form(...),
#     question_index: int = Form(...),
#     answer: str = Form(...)
# ):
#     """提交面试问题的答案"""
#     if interview_id not in interviews_db:
#         raise HTTPException(status_code=404, detail="面试记录不存在")
    
#     # 确保问题索引有效
#     if question_index >= len(interviews_db[interview_id]["questions"]):
#         raise HTTPException(status_code=400, detail="无效的问题索引")
    
#     # 存储答案
#     if "answers" not in interviews_db[interview_id]:
#         interviews_db[interview_id]["answers"] = {}
    
#     interviews_db[interview_id]["answers"][str(question_index)] = {
#         "answer": answer,
#         "submitted_at": datetime.now().isoformat()
#     }
    
#     return JSONResponse({
#         "success": True,
#         "message": "答案已提交"
#     })

# @app.post("/api/generate-report")
# async def generate_report(interview_id: str = Form(...)):
#     """生成面试报告"""
#     if interview_id not in interviews_db:
#         raise HTTPException(status_code=404, detail="面试记录不存在")
    
#     # 模拟报告生成过程
#     report_id = str(uuid.uuid4())
#     report_path = f"{config.REPORT_DIR}/{report_id}.pdf"
    
#     # 在实际应用中，这里会生成PDF报告
#     # 现在我们只是创建一个空文件作为示例
#     Path(report_path).touch()
    
#     # 存储报告信息
#     reports_db[report_id] = {
#         "interview_id": interview_id,
#         "report_path": report_path,
#         "created_at": datetime.now().isoformat()
#     }
    
#     interviews_db[interview_id]["status"] = "completed"
#     interviews_db[interview_id]["report_id"] = report_id
    
#     return JSONResponse({
#         "success": True,
#         "report_id": report_id,
#         "message": "报告生成完成"
#     })

# @app.get("/api/download-report/{report_id}")
# async def download_report(report_id: str):
#     """下载面试报告"""
#     if report_id not in reports_db:
#         raise HTTPException(status_code=404, detail="报告不存在")
    
#     report_path = reports_db[report_id]["report_path"]
    
#     if not os.path.exists(report_path):
#         raise HTTPException(status_code=404, detail="报告文件不存在")
    
#     return FileResponse(
#         report_path,
#         media_type="application/pdf",
#         filename=f"AI面试报告_{report_id}.pdf"
#     )

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
   
   
def generate_sample_questions(job_description: str) -> list:
    """根据职位描述生成示例问题"""
    # 在实际应用中，这里会调用AI模型生成问题
    # 现在我们返回一些通用问题
    return [
        "请介绍一下您在前端开发方面的经验？",
        "您如何处理与UI设计师的意见分歧？",
        "请解释一下React Hooks的工作原理？",
        "您如何优化前端性能？",
        "为什么选择我们公司？"
    ]



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        # reload=config.DEBUG,
        # workers=config.WORKERS
    )