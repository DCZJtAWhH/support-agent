"""定义飞轮后台及反馈的同源公开接口。"""

from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()
WEB_DIRECTORY = Path(__file__).resolve().parents[1] / "web"


class FeedbackSubmission(BaseModel):
    """仅接收轮次身份与选择，不信任客户端提交的用户问题或召回片段。"""

    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    request_id: UUID
    value: Literal["up", "down"]


class ReviewDecision(BaseModel):
    """只允许人工通过或驳回，核准答案另由服务校验非空。"""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["approved", "rejected"]
    approved_answer: str = Field(default="", max_length=6000)


@router.get("/reviews", response_class=FileResponse, include_in_schema=False)
async def review_page():
    """提供独立知识审核与运营页面。"""
    return FileResponse(WEB_DIRECTORY / "review.html", headers={"Cache-Control": "no-cache"})


@router.post("/api/feedback")
async def feedback(body: FeedbackSubmission, request: Request):
    """保存已完成轮次的反馈，未解决的问题进入统一问题池。"""
    try:
        return await request.app.state.flywheel.feedback(
            str(body.conversation_id),
            str(body.request_id),
            body.value,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/review-queue")
async def review_queue(
    request: Request, status: Literal["", "pending", "publishing", "approved", "rejected"] = ""
):
    """按审核状态返回去重后的知识缺口。"""
    return await request.app.state.flywheel.list_reviews(status or None)


@router.get("/api/review-queue/{review_id}")
async def review_detail(review_id: int, request: Request):
    """展开一个知识缺口对应的全部用户原话及召回快照。"""
    try:
        return await request.app.state.flywheel.review_details(review_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/flywheel/process")
async def process_pool(request: Request):
    """使用与自动扫描相同的流水线，手动整理尚未归并的问题。"""
    try:
        return {"processed": await request.app.state.flywheel.process_pending()}
    except Exception as exc:
        raise HTTPException(502, "问题整理失败，原问题仍保留在问题池，可稍后重试。") from exc


@router.post("/api/review-queue/{review_id}/decision")
async def decide_review(review_id: int, body: ReviewDecision, request: Request):
    """保存人工核准或驳回；通过时等待知识索引同步成功再报告可检索。"""
    try:
        return await request.app.state.flywheel.decide(
            review_id,
            body.decision,
            body.approved_answer,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "知识尚未完成同步，请保留原核准答案重试审核。") from exc
