# -*- coding: utf-8 -*-
"""
声见其迹 (SONOTRACE) — 分析历史与人工复核队列持久化层

系统里的「不确定性拒识」机制, 会在 MC-Dropout 认知不确定度 ≥ 拒识阈值、或六维
前置质控未通过时输出 verdict_key='pending' (建议人工复核)。本模块把这类复核
队列与每次分析的完整结果落盘, 让医生/评委能回看历史 —— 这是比赛展示端
"可追溯" 与 "风险-覆盖率机制" 的落地点。

数据文件: BASE_DIR/data/sonotrace.db  (目录不存在时自动创建)

设计原则 (答辩现场零翻车优先于数据完整性):
  * 只用标准库 sqlite3, 不引入 ORM / 新依赖
  * 每次操作新建连接 + ``with`` 事务 + 用完即关; 不跨线程共享 connection,
    SQLite 的多线程安全靠这个保证 (WAL + busy_timeout 容忍并发写)
  * 除 ``resolve_review`` 的非法 status 抛 ValueError 外, 任何数据库错误一律
    logging.warning 后返回安全空值 (None / {} / 0 / []), 绝不抛给调用方
  * verdict_key == 'pending' 时自动生成一条 status='open' 的复核队列项

表结构:
  analyses  单次分析的结构化摘要 + 完整结果 JSON payload (体积可能较大)
  reviews   复核队列 (open / confirmed / rejected / escalated)
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta

from . import config as C

# ---------------------------------------------------------------- 常量
LOGGER = logging.getLogger(__name__)

DATA_DIR = os.path.join(C.BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "sonotrace.db")

VERDICT_KEYS = ("normal", "abnormal", "pending")
REVIEW_STATUSES = ("open", "confirmed", "rejected", "escalated")
# 只有这三个是"处理完成"的状态; 'open' 由系统自动生成, 不允许通过 resolve_review 写回
RESOLVED_STATUSES = ("confirmed", "rejected", "escalated")

_MAX_LIMIT = 1000          # 单页条数上限, 防止前端误传 limit=10**9 拖垮接口
_DEFAULT_LIMIT = 50

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,          -- ISO8601 本地时间
    filename      TEXT NOT NULL,
    sha256        TEXT,                   -- 音频内容哈希, 用于去重/溯源
    duration_sec  REAL,
    prob_abnormal REAL,
    uncertainty   REAL,
    verdict_key   TEXT,                   -- 'normal' | 'abnormal' | 'pending'
    verdict_label TEXT,
    quality_pass  INTEGER,                -- 0/1
    engine_mode   TEXT,
    abstain_thr   REAL,
    payload_json  TEXT                    -- 完整分析结果的 JSON (可能很大)
);

CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id  INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'open',   -- 'open' | 'confirmed' | 'rejected' | 'escalated'
    reviewer     TEXT,
    note         TEXT,
    resolved_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_sha256     ON analyses(sha256);
CREATE INDEX IF NOT EXISTS idx_reviews_status      ON reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_analysis_id ON reviews(analysis_id);
"""

_INSERT_ANALYSIS_SQL = """
INSERT INTO analyses (created_at, filename, sha256, duration_sec, prob_abnormal,
                      uncertainty, verdict_key, verdict_label, quality_pass,
                      engine_mode, abstain_thr, payload_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INIT_DONE = False         # 惰性建表标记; 首次访问自动建库, 之后的重复调用直接返回


# ==================================================================== 内部工具
def _now() -> str:
    """当前本地时间, ISO8601 秒级精度 (同一格式保证 created_at 可按字符串比较)。"""
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir() -> None:
    """确保 data/ 目录存在 (幂等)。"""
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


@contextlib.contextmanager
def _conn():
    """
    打开一个短连接并包成事务上下文。

    退出时: 正常 -> 提交; 异常 -> 回滚; 无论哪种都关闭连接。
    WAL 让"写不阻塞读", busy_timeout 吸收并发写冲突, foreign_keys 保证
    reviews 能随 analyses 级联删除。
    """
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:                                        # 事务作用域
            yield conn
    finally:
        conn.close()


def _as_float(value) -> float | None:
    """安全转 float; 缺失 / 非数值 / NaN-inf 一律返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):   # NaN / inf 不入库
        return None
    return out


def _as_text(value) -> str | None:
    """安全转 str; 空串与 None 统一返回 None。"""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def _page(limit, offset) -> tuple[int, int]:
    """规范化分页参数, 非法输入退回默认值, 再夹到安全区间。"""
    try:
        lim = int(limit)
    except (TypeError, ValueError):
        lim = _DEFAULT_LIMIT
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    return max(0, min(lim, _MAX_LIMIT)), max(0, off)


def _json_default(obj):
    """
    payload 兜底序列化: pipeline 结果里可能混入 numpy 标量/数组等非原生类型,
    这里转成可 JSON 的类型, 保证 payload_json 永远不会因为序列化失败而丢记录。
    """
    for attr in ("tolist", "item"):                       # numpy 数组 / numpy 标量
        func = getattr(obj, attr, None)
        if callable(func):
            try:
                return func()
            except Exception:                             # noqa: BLE001
                pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace")
    return str(obj)


# ==================================================================== 建表
def init_db() -> None:
    """幂等建表。首次调用创建目录和表, 重复调用无副作用。"""
    global _INIT_DONE
    try:
        _ensure_dir()
        with _conn() as conn:
            conn.executescript(_SCHEMA_SQL)
        _INIT_DONE = True
    except Exception as ex:                                # noqa: BLE001
        LOGGER.warning("init_db 失败 (将降级为无历史模式): %s: %s", type(ex).__name__, ex)


def _ensure_ready() -> None:
    """
    惰性初始化: 任何入口在第一次被调用时自动建库建表。
    即使 app.py 忘记显式调用 init_db, 历史/复核接口也能正常工作 (幂等)。
    """
    if _INIT_DONE:
        return
    init_db()


# ==================================================================== 写入
def save_analysis(result: dict, audio_sha256: str | None = None) -> int:
    """
    把 pipeline.analyze() 返回的完整 dict 存一条记录, 返回 analyses.id。

    所有字段都用 .get 链安全取值 —— result 里任何键缺失都不会 KeyError:
      result['filename'] / ['duration_sec'] / ['engine_mode']
      result['quality']['passed']
      result['prediction']['prob_abnormal'|'uncertainty'|'verdict_key'|'label'|'abstain_threshold']
    verdict_key == 'pending' 时自动插一条 reviews 记录 (status='open')。
    写入失败只打印警告并返回 0 (调用方据 0 判断"本次未落盘")。
    """
    _ensure_ready()
    try:
        if not isinstance(result, dict):
            LOGGER.warning("save_analysis: result 不是 dict, 跳过本次写入")
            return 0

        quality = result.get("quality")
        prediction = result.get("prediction")
        quality = quality if isinstance(quality, dict) else {}
        prediction = prediction if isinstance(prediction, dict) else {}

        filename = _as_text(result.get("filename")) or "unknown.wav"
        duration_sec = _as_float(result.get("duration_sec"))
        engine_mode = _as_text(result.get("engine_mode"))
        # 缺字段/非布尔值一律按"未通过"处理: 宁可多复核, 不可漏判
        quality_pass = 1 if quality.get("passed") else 0
        prob_abnormal = _as_float(prediction.get("prob_abnormal"))
        uncertainty = _as_float(prediction.get("uncertainty"))
        # 缺失判定键时按待复核处理 (保守策略)
        verdict_key = _as_text(prediction.get("verdict_key")) or "pending"
        verdict_label = _as_text(prediction.get("label"))
        abstain_thr = _as_float(prediction.get("abstain_threshold"))

        try:
            payload_json = json.dumps(result, ensure_ascii=False, default=_json_default)
        except Exception as ex:                           # noqa: BLE001
            # payload 兜底仍未成功: 只丢 payload, 保住摘要字段
            LOGGER.warning("save_analysis: payload 序列化失败, 仅保存摘要: %s: %s",
                           type(ex).__name__, ex)
            payload_json = None

        now = _now()
        with _conn() as conn:
            cur = conn.execute(_INSERT_ANALYSIS_SQL, (
                now, filename, _as_text(audio_sha256), duration_sec, prob_abnormal,
                uncertainty, verdict_key, verdict_label, quality_pass,
                engine_mode, abstain_thr, payload_json,
            ))
            analysis_id = int(cur.lastrowid or 0)
            # 拒识样本自动进复核队列 (与 analyses 同事务, 不会只写半条)
            if analysis_id > 0 and verdict_key == "pending":
                conn.execute(
                    "INSERT INTO reviews (analysis_id, created_at, status) "
                    "VALUES (?, ?, 'open')",
                    (analysis_id, now),
                )
        return analysis_id
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("save_analysis 失败: %s: %s", type(ex).__name__, ex)
        return 0


# ==================================================================== 读取
def list_analyses(limit: int = 50, offset: int = 0,
                  verdict_key: str | None = None) -> dict:
    """
    分页列出分析历史 (按 id 倒序, 新的在前)。

    返回 {'total': int, 'items': [...]}; items 不含 payload_json (体积太大,
    需要完整结果请用 get_analysis)。has_review 表示该条是否已进过复核队列。
    """
    _ensure_ready()
    try:
        lim, off = _page(limit, offset)
        vk = _as_text(verdict_key)
        where, args = "", []
        if vk:
            where = "WHERE a.verdict_key = ?"
            args.append(vk)

        sql = f"""
            SELECT a.id, a.created_at, a.filename, a.duration_sec, a.prob_abnormal,
                   a.uncertainty, a.verdict_key, a.verdict_label, a.quality_pass,
                   a.engine_mode,
                   EXISTS (SELECT 1 FROM reviews r WHERE r.analysis_id = a.id) AS has_review
              FROM analyses a
              {where}
             ORDER BY a.id DESC
             LIMIT ? OFFSET ?
        """
        with _conn() as conn:
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM analyses a {where}", args).fetchone()[0])
            rows = conn.execute(sql, args + [lim, off]).fetchall()

        items = [{
            "id": int(r["id"]),
            "created_at": r["created_at"],
            "filename": r["filename"],
            "duration_sec": r["duration_sec"],
            "prob_abnormal": r["prob_abnormal"],
            "uncertainty": r["uncertainty"],
            "verdict_key": r["verdict_key"],
            "verdict_label": r["verdict_label"],
            "quality_pass": bool(r["quality_pass"]),
            "engine_mode": r["engine_mode"],
            "has_review": bool(r["has_review"]),
        } for r in rows]
        return {"total": total, "items": items}
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("list_analyses 失败: %s: %s", type(ex).__name__, ex)
        return {"total": 0, "items": []}


def get_analysis(analysis_id: int) -> dict | None:
    """
    返回单条完整记录, 并把 payload_json 解析回 dict 放在 'payload' 键下。
    记录不存在 / id 非法 / 数据库不可用时返回 None。
    """
    _ensure_ready()
    try:
        aid = int(analysis_id)
    except (TypeError, ValueError):
        return None
    try:
        sql = """
            SELECT a.*,
                   EXISTS (SELECT 1 FROM reviews r WHERE r.analysis_id = a.id) AS has_review,
                   (SELECT COUNT(*) FROM reviews r WHERE r.analysis_id = a.id) AS review_count
              FROM analyses a
             WHERE a.id = ?
        """
        with _conn() as conn:
            row = conn.execute(sql, (aid,)).fetchone()
        if row is None:
            return None

        record = dict(row)
        raw_payload = record.pop("payload_json", None)
        payload = None
        if raw_payload:
            try:
                payload = json.loads(raw_payload)
            except Exception as ex:                       # noqa: BLE001
                LOGGER.warning("get_analysis: payload 解析失败 (id=%s): %s: %s",
                               aid, type(ex).__name__, ex)
        record["payload"] = payload
        record["has_review"] = bool(record.get("has_review"))
        record["quality_pass"] = bool(record.get("quality_pass"))
        return record
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("get_analysis 失败: %s: %s", type(ex).__name__, ex)
        return None


def list_reviews(status: str = "open", limit: int = 50, offset: int = 0) -> dict:
    """
    分页列出复核队列 (按 id 倒序), 每条 join analyses 带出判读所需的上下文字段:
        filename / prob_abnormal / uncertainty / verdict_label / created_at

    status 传 None 或空串表示不过滤 (全部状态)。返回 {'total': int, 'items': [...]}。
    """
    _ensure_ready()
    try:
        lim, off = _page(limit, offset)
        st = _as_text(status)
        where, args = "", []
        if st:
            where = "WHERE r.status = ?"
            args.append(st)

        sql = f"""
            SELECT r.id, r.analysis_id, r.created_at, r.status, r.reviewer, r.note,
                   r.resolved_at,
                   a.filename, a.prob_abnormal, a.uncertainty, a.verdict_label,
                   a.verdict_key AS analysis_verdict_key,
                   a.quality_pass AS analysis_quality_pass,
                   a.created_at  AS analysis_created_at
              FROM reviews r
              LEFT JOIN analyses a ON a.id = r.analysis_id
              {where}
             ORDER BY r.id DESC
             LIMIT ? OFFSET ?
        """
        with _conn() as conn:
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM reviews r {where}", args).fetchone()[0])
            rows = conn.execute(sql, args + [lim, off]).fetchall()

        items = [{
            "id": int(r["id"]),
            "analysis_id": int(r["analysis_id"]),
            "created_at": r["created_at"],
            "status": r["status"],
            "reviewer": r["reviewer"],
            "note": r["note"],
            "resolved_at": r["resolved_at"],
            "filename": r["filename"],
            "prob_abnormal": r["prob_abnormal"],
            "uncertainty": r["uncertainty"],
            "verdict_label": r["verdict_label"],
            "verdict_key": r["analysis_verdict_key"],
            "quality_pass": bool(r["analysis_quality_pass"]),
            "analysis_created_at": r["analysis_created_at"],
        } for r in rows]
        return {"total": total, "items": items}
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("list_reviews 失败: %s: %s", type(ex).__name__, ex)
        return {"total": 0, "items": []}


# ==================================================================== 复核
def resolve_review(review_id: int, status: str, reviewer: str = "clinician",
                   note: str = "") -> bool:
    """
    填写复核结论: 写入 status/reviewer/note/resolved_at。

    status 必须是 {'confirmed','rejected','escalated'} 之一, 否则抛 ValueError
    (这是本模块唯一会向调用方抛出的异常, 属于调用方编程错误而非运行时故障)。
    返回是否真的更新到一行; 数据库故障时打印警告并返回 False。
    """
    if status not in RESOLVED_STATUSES:
        raise ValueError(f"status 必须是 {RESOLVED_STATUSES} 之一, 收到: {status!r}")
    _ensure_ready()
    try:
        rid = int(review_id)
    except (TypeError, ValueError):
        LOGGER.warning("resolve_review: review_id 非法: %r", review_id)
        return False
    try:
        sql = ("UPDATE reviews SET status = ?, reviewer = ?, note = ?, resolved_at = ? "
               "WHERE id = ?")
        with _conn() as conn:
            cur = conn.execute(sql, (status, _as_text(reviewer),
                                     _as_text(note), _now(), rid))
            updated = int(cur.rowcount or 0)
        return updated > 0
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("resolve_review 失败: %s: %s", type(ex).__name__, ex)
        return False


# ==================================================================== 统计 / 清理
def stats() -> dict:
    """
    复核看板用的聚合指标:
      total_analyses    累计分析条数
      open_reviews      仍待人工复核的条数
      by_verdict        {'normal': n, 'abnormal': n, 'pending': n} (三类恒在, 无记录为 0)
      mean_uncertainty  平均 MC-Dropout 不确定度 (无样本时为 None)
      abstain_rate      待复核(pending)占比, 含"质控未过"与"不确定拒识"两类;
                        无样本时为 None
    """
    _ensure_ready()
    try:
        by_verdict = {k: 0 for k in VERDICT_KEYS}
        with _conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(uncertainty) AS mean_unc FROM analyses"
            ).fetchone()
            total = int(row["n"] or 0)
            mean_unc = _as_float(row["mean_unc"])

            for r in conn.execute(
                    "SELECT verdict_key, COUNT(*) AS n FROM analyses GROUP BY verdict_key"):
                key = _as_text(r["verdict_key"]) or "unknown"
                by_verdict[key] = int(r["n"] or 0)

            open_reviews = int(conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE status = 'open'").fetchone()[0] or 0)

        abstain_rate = None
        if total > 0:
            abstain_rate = round(by_verdict.get("pending", 0) / float(total), 4)
        return {
            "total_analyses": total,
            "open_reviews": open_reviews,
            "by_verdict": by_verdict,
            "mean_uncertainty": round(mean_unc, 4) if mean_unc is not None else None,
            "abstain_rate": abstain_rate,
        }
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("stats 失败: %s: %s", type(ex).__name__, ex)
        return {}


def prune(keep_days: int = 30) -> int:
    """
    删除 created_at 早于 keep_days 天的 analyses, 返回删除行数。

    对应的 reviews 由外键 ON DELETE CASCADE 一并清除 (连接上已开
    PRAGMA foreign_keys=ON)。数据库故障时返回 0。
    """
    _ensure_ready()
    try:
        days = int(keep_days)
    except (TypeError, ValueError):
        days = 30
    days = max(0, days)
    try:
        # created_at 统一为 ISO8601 秒级本地时间, 因此字符串比较等价于时间先后
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with _conn() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE created_at < ?", (cutoff,))
            removed = int(cur.rowcount or 0)
        return removed
    except Exception as ex:                               # noqa: BLE001
        LOGGER.warning("prune 失败: %s: %s", type(ex).__name__, ex)
        return 0