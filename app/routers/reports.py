from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List

from app.database import get_session
from app.models.weekly_report import WeeklyReport
from app.services.reports.weekly_compiler import compile_weekly_report

router = APIRouter(prefix="/reports", tags=["Weekly Reports"])

@router.get("/weekly", response_model=List[WeeklyReport])
def list_weekly_reports(session: Session = Depends(get_session)):
    """Retrieve all compiled weekly reports."""
    statement = select(WeeklyReport).order_by(WeeklyReport.created_at.desc())
    return session.exec(statement).all()

@router.get("/weekly/{report_id}", response_model=WeeklyReport)
def get_weekly_report(report_id: int, session: Session = Depends(get_session)):
    """Retrieve details of a specific weekly report."""
    report = session.get(WeeklyReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Weekly report not found.")
    return report

@router.post("/weekly/compile", response_model=WeeklyReport)
def manual_compile_weekly_report(session: Session = Depends(get_session)):
    """Manually trigger compilation of the Weekly AI & Developer Intelligence Report."""
    report = compile_weekly_report(session)
    if not report:
        raise HTTPException(status_code=500, detail="Failed to compile weekly report due to insufficient data or model error.")
    return report
