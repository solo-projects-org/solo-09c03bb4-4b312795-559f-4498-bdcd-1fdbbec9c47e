"""Concurrency guarantees: serialized writers and snapshot reads.

Uses only the stdlib (threads + sqlite3) so it runs without FastAPI.
"""

import tempfile
import threading
import unittest
from pathlib import Path

from app.db import connect, init_db
from app.service import (
    ConflictError,
    IngestionError,
    UnresolvableTimelineError,
    ingest_batch,
    query_traffic,
)
from app.timeline import Record


def r(sid: int, local: str, count: int) -> Record:
    return Record(sid, local, count)


SEED = [
    r(100, "2024-11-03T00:45:00", 120),
    r(128, "2024-11-03T01:20:00", 340),
]


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        init_db(connect(self.db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_racing_conflicting_batches_never_both_land(self):
        # Four threads contend for the same sample id with different values.
        # BEGIN IMMEDIATE serializes them; exactly one value can ever land.
        barrier = threading.Barrier(4)
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker(count: int):
            conn = connect(self.db_path)
            barrier.wait()  # maximize lock contention
            try:
                ingest_batch(
                    conn,
                    [r(128, "2024-11-03T01:20:00", count)],
                )
                kind = "accepted"
            except IngestionError:
                kind = "rejected"
            finally:
                conn.close()
            with lock:
                outcomes.append(kind)

        threads = [
            threading.Thread(target=worker, args=(count,), daemon=True)
            for count in (340, 999, 777, 555)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(sorted(outcomes), ["accepted", "rejected",
                                            "rejected", "rejected"])
        conn = connect(self.db_path)
        rows = conn.execute(
            "SELECT sample_id, pass_count FROM records"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 128)
        self.assertIn(rows[0][1], (340, 999, 777, 555))
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT status FROM batch_audit ORDER BY id"
            ).fetchall()
        ]
        self.assertEqual(statuses.count("accepted"), 1)
        self.assertEqual(statuses.count("rejected"), 3)
        # every decision has exactly one audit row: nothing silently un-audited
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM batch_audit").fetchone()[0],
            4,
        )
        conn.close()

    def test_parallel_batches_leave_a_legal_warehouse(self):
        writer = connect(self.db_path)
        ingest_batch(writer, SEED)
        writer.close()

        barrier = threading.Barrier(5)
        lock = threading.Lock()
        kinds: list[str] = []

        def worker(payload):
            conn = connect(self.db_path)
            barrier.wait()
            try:
                ingest_batch(conn, payload)
                kind = "accepted"
            except ConflictError:
                kind = "conflict"
            except UnresolvableTimelineError:
                kind = "illegal"
            finally:
                conn.close()
            with lock:
                kinds.append(kind)

        threads = [
            # idempotent duplicate of a seeded row
            threading.Thread(target=worker, args=([
                r(128, "2024-11-03T01:20:00", 340)],), daemon=True),
            # conflicting value for a seeded id -> always rejected
            threading.Thread(target=worker, args=([
                r(128, "2024-11-03T01:20:00", 999)],), daemon=True),
            # fresh ids far outside the repeated hour -> always accepted
            threading.Thread(target=worker, args=([
                r(200, "2024-11-03T04:00:00", 10),
                r(201, "2024-11-03T04:05:00", 11)],), daemon=True),
            threading.Thread(target=worker, args=([
                r(202, "2024-11-03T04:10:00", 12),
                r(203, "2024-11-03T04:15:00", 13)],), daemon=True),
            # illegal on its own merits -> always rejected
            threading.Thread(target=worker, args=([
                r(5, "2024-11-03T03:00:00", 50),
                r(15, "2024-11-03T00:30:00", 60)],), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(kinds.count("conflict"), 1)
        self.assertEqual(kinds.count("illegal"), 1)
        self.assertEqual(kinds.count("accepted"), 3)

        conn = connect(self.db_path)
        # invariant 1: the whole warehouse still admits a legal timeline
        view = query_traffic(conn)
        self.assertNotIn("error", view)
        # invariant 2: seeded values were never overwritten
        stored = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT sample_id, pass_count FROM records"
            ).fetchall()
        }
        self.assertEqual(stored[128], 340)
        self.assertEqual(stored[100], 120)
        self.assertEqual(stored[200], 10)
        conn.close()

    def test_reads_never_mix_old_verdicts_with_new_rows(self):
        writer = connect(self.db_path)
        ingest_batch(writer, SEED)

        views: list[int] = []
        stop = threading.Event()

        def reader():
            conn = connect(self.db_path)
            try:
                while not stop.is_set():
                    view = query_traffic(conn)
                    detail_ids = {d["sample_id"] for d in view["details"]}
                    pending_ids = set(view["pending_sample_ids"])
                    # snapshot invariant: complete partition, no overlap/mix-up
                    self.assertTrue(detail_ids.isdisjoint(pending_ids))
                    self.assertEqual(
                        view["total_records"],
                        view["determined_count"] + view["pending_count"],
                    )
                    self.assertEqual(len(detail_ids), view["determined_count"])
                    self.assertEqual(len(pending_ids), view["pending_count"])
                    views.append(view["total_records"])
            finally:
                conn.close()

        readers = [
            threading.Thread(target=reader, daemon=True) for _ in range(3)
        ]
        for t in readers:
            t.start()

        try:
            for sid in range(200, 210):
                # distinct wall times -> distinct UTC, always a legal extension
                minute = sid - 200
                ingest_batch(
                    writer,
                    [r(sid, f"2024-11-03T04:{minute:02d}:00", sid)],
                )
        finally:
            stop.set()
            for t in readers:
                t.join(timeout=30)
        writer.close()

        # every observed view shows a whole-warehouse size (never a partial mix)
        self.assertTrue(set(views).issubset(set(range(2, 13))))
        self.assertGreater(len(views), 0)

    def test_audit_order_matches_decision_effectiveness_under_contention(self):
        # Mixed accept/reject batches race through the same write lock.
        # The audit trail must contain one row per decision in the exact order
        # the serialized transactions became effective (gapless ids).  Each
        # pair targets a fresh sample id, so both outcomes are always valid
        # regardless of which thread wins the lock first.
        jobs = [
            # (payload, result if it reaches the lock first, result if second)
            ([r(10, "2024-11-03T00:10:00", 1)], {"accepted", "rejected"}),
            ([r(10, "2024-11-03T00:10:00", 2)], {"accepted", "rejected"}),
            ([r(20, "2024-11-03T00:20:00", 3)], {"accepted", "rejected"}),
            ([r(20, "2024-11-03T00:20:00", 4)], {"accepted", "rejected"}),
            ([r(30, "2024-11-03T00:30:00", 5)], {"accepted", "rejected"}),
            ([r(30, "2024-11-03T00:30:00", 6)], {"accepted", "rejected"}),
        ]
        barrier = threading.Barrier(len(jobs))
        lock = threading.Lock()
        errors: list[BaseException] = []

        def runner(payload, allowed):
            barrier.wait()
            conn = connect(self.db_path)
            try:
                try:
                    ingest_batch(conn, payload)
                    kind = "accepted"
                except ConflictError:
                    kind = "rejected"
                if kind not in allowed:  # pragma: no cover - failure path
                    raise AssertionError(f"unexpected outcome {kind}")
            except BaseException as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)
            finally:
                conn.close()

        threads = [
            threading.Thread(target=runner, args=job, daemon=True) for job in jobs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [])

        conn = connect(self.db_path)
        rows = conn.execute(
            "SELECT id, status FROM batch_audit ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 6)
        self.assertEqual([row[0] for row in rows], [1, 2, 3, 4, 5, 6])
        statuses = [row[1] for row in rows]
        # exactly one winner and one rejection per contended sample id
        self.assertEqual(statuses.count("accepted"), 3)
        self.assertEqual(statuses.count("rejected"), 3)
        # rejected rows carry their durable reason
        reasons = [
            row[0]
            for row in conn.execute(
                "SELECT reason FROM batch_audit WHERE status = 'rejected'"
            ).fetchall()
        ]
        self.assertEqual(len(reasons), 3)
        self.assertTrue(all("already stored with different content" in x
                            for x in reasons))
        # only the three accepted readings exist
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM records").fetchone()[0], 3
        )
        conn.close()


if __name__ == "__main__":
    unittest.main()
