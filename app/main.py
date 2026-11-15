"""FastAPI application: DST-safe airport traffic ingestion.

Endpoints
---------
GET  /health         liveness probe
POST /upload         back-fill one CSV batch (atomic accept / whole-batch reject)
GET  /traffic        determined UTC details + pending sample ids
GET  /batches        batch audit trail in decision-effectiveness order
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from .csv_utils import CsvValidationError, parse_counter_csv
from .db import connect, init_db
from .service import (
    AuditPersistError,
    IngestionError,
    ingest_batch,
    list_audit,
    query_traffic,
)
from .timeline import Record


def get_db() -> Iterator[sqlite3.Connection]:
    """Yield one connection per request and always close it.

    A per-request connection is safe under any server thread pool and under
    multiple workers: SQLite itself (WAL + BEGIN IMMEDIATE + busy_timeout)
    coordinates cross-connection concurrency.
    """

    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Schema bootstrap only; request handlers open their own connections.
    conn = connect()
    try:
        init_db(conn)
    finally:
        conn.close()
    yield


app = FastAPI(
    title="机场入口客流 DST 回拨入仓服务",
    description=(
        "接收 America/New_York 本地时刻（无偏移）的补传 CSV。回拨当天 "
        "01:00–02:00 的钟面时刻对应两个 UTC，本服务只发布在所有合法时间线中 "
        "UTC 都唯一的记录，其余列为待定采样号。批次读数与受理审计在同一事务内 "
        "提交：受理必有审计，拒收原因持久留存且不留任何新增读数。"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(IngestionError)
async def _ingestion_error_handler(_: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"accepted": False, "error": exc.reason},
    )


@app.exception_handler(AuditPersistError)
async def _audit_error_handler(_: Request, exc: AuditPersistError) -> JSONResponse:
    # Never pretend the decision was recorded: the transaction was rolled back.
    return JSONResponse(
        status_code=exc.status_code,
        content={"accepted": False, "error": exc.reason},
    )


@app.exception_handler(CsvValidationError)
async def _csv_error_handler(_: Request, exc: CsvValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"accepted": False, "error": f"invalid CSV: {exc}"},
    )


@app.get("/health", tags=["运维"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/upload",
    tags=["补传"],
    summary="补传一批 CSV（采样号, 本地时刻, 通过人数）",
    responses={
        200: {"description": "读数与受理审计在同一事务提交；同值重传为幂等受理"},
        400: {"description": "CSV 格式错误"},
        409: {"description": "同号异值冲突，整批拒收（拒收原因已持久化）"},
        422: {"description": "合并后排不出合法 UTC 时间线，整批拒收（原因已持久化）"},
        500: {"description": "审计写入失败，事务已回滚，决议未留痕"},
    },
)
async def upload(
    file: UploadFile = File(..., description="UTF-8 CSV 文件"),
    db=Depends(get_db),
) -> dict:
    content = await file.read()
    records: list[Record] = parse_counter_csv(content)
    result = ingest_batch(db, records)
    return {
        "accepted": True,
        "batch_ref": result.batch_ref,
        "record_count": result.record_count,
        "inserted_sample_ids": result.inserted,
        "skipped_duplicate_sample_ids": result.skipped_duplicates,
    }


@app.get(
    "/traffic",
    tags=["查询"],
    summary="查询客流明细（UTC 唯一的记录）与待定采样号",
)
def traffic(db=Depends(get_db)) -> dict:
    return query_traffic(db)


@app.get(
    "/batches",
    tags=["运维"],
    summary="批次受理审计（含拒收原因，按决议生效顺序排列）",
)
def batches(db=Depends(get_db)) -> dict:
    return {"batches": list_audit(db)}
