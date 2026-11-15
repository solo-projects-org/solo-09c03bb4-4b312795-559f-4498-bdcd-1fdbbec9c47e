"""SQLite-backed service tests: atomic batches, conflicts, idempotency, audit."""

import tempfile
import unittest
from pathlib import Path

from app.db import connect, init_db
from app.service import (
    AuditPersistError,
    ConflictError,
    IngestionError,
    UnresolvableTimelineError,
    ingest_batch,
    list_audit,
    query_traffic,
)
from app.timeline import Record
from app.csv_utils import CsvValidationError, parse_counter_csv


def r(sid: int, local: str, count: int) -> Record:
    return Record(sid, local, count)


INITIAL = [
    r(100, "2024-11-03T00:45:00", 120),
    r(128, "2024-11-03T01:20:00", 340),
    r(148, "2024-11-03T01:55:00", 300),
    r(160, "2024-11-03T02:15:00", 260),
    r(180, "2024-11-03T03:10:00", 210),
]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db_path)
        init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _ingest(self, records):
        return ingest_batch(self.conn, records)

    def _audit_statuses(self):
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT status FROM batch_audit ORDER BY id"
            ).fetchall()
        ]

    def _record_count(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]

    def test_full_backfill_scenario(self):
        self._ingest(INITIAL)
        view = query_traffic(self.conn)
        self.assertEqual(view["total_records"], 5)
        self.assertEqual(view["determined_count"], 3)
        self.assertEqual(view["pending_sample_ids"], [128, 148])
        self.assertEqual(
            [d["sample_id"] for d in view["details"]], [100, 160, 180]
        )
        # pending payload exposes the two possible UTC instants
        pending_by_id = {p["sample_id"]: p for p in view["pending"]}
        self.assertEqual(
            pending_by_id[128]["possible_utc_earliest"],
            "2024-11-03T05:20:00+00:00",
        )
        self.assertEqual(
            pending_by_id[128]["possible_utc_latest"],
            "2024-11-03T06:20:00+00:00",
        )

        # one same-wall backfill row disambiguates the whole repeated hour
        self._ingest([r(127, "2024-11-03T01:20:00", 84)])
        view = query_traffic(self.conn)
        self.assertEqual(view["pending_sample_ids"], [])
        self.assertEqual(view["determined_count"], 6)
        by_id = {d["sample_id"]: d for d in view["details"]}
        self.assertEqual(by_id[127]["utc_time"], "2024-11-03T05:20:00+00:00")
        self.assertEqual(by_id[128]["utc_time"], "2024-11-03T06:20:00+00:00")
        self.assertEqual(by_id[148]["utc_time"], "2024-11-03T06:55:00+00:00")

    def test_same_id_same_content_is_idempotent(self):
        batch = [r(128, "2024-11-03T01:20:00", 340)]
        first = self._ingest(batch)
        self.assertEqual(first.inserted, [128])
        second = self._ingest(batch)
        self.assertEqual(second.inserted, [])
        self.assertEqual(second.skipped_duplicates, [128])
        self.assertEqual(self._record_count(), 1)
        # a same-value re-upload is still a formal acceptance: both decisions
        # have their own durable acceptance audit row
        self.assertEqual(self._audit_statuses(), ["accepted", "accepted"])

    def test_same_id_different_value_rejects_whole_batch(self):
        self._ingest([r(128, "2024-11-03T01:20:00", 340)])
        with self.assertRaises(ConflictError):
            self._ingest([
                r(128, "2024-11-03T01:20:00", 999),
                r(199, "2024-11-03T04:00:00", 10),
            ])
        # neither row of the rejected batch landed
        self.assertEqual(self._record_count(), 1)
        # the rejection itself is audited with its reason persisted
        self.assertEqual(self._audit_statuses(), ["accepted", "rejected"])
        rejected = self.conn.execute(
            "SELECT reason FROM batch_audit WHERE status = 'rejected'"
        ).fetchone()
        self.assertIn("sample_id 128", rejected[0])

    def test_illegal_timeline_rejects_whole_batch(self):
        self._ingest([r(100, "2024-11-03T00:45:00", 120)])
        with self.assertRaises(UnresolvableTimelineError):
            self._ingest([
                r(5, "2024-11-03T03:00:00", 50),
                r(15, "2024-11-03T00:30:00", 60),
            ])
        self.assertEqual(self._record_count(), 1)
        # rejection reason persists even though the business rows rolled back
        self.assertEqual(self._audit_statuses(), ["accepted", "rejected"])

    def test_backfill_infeasible_against_stored_data_is_rolled_back(self):
        self._ingest(INITIAL)
        # Stored 128 plus two more readings at the same ambiguous wall time
        # gives three samples with only two UTC slots -> impossible ordering.
        with self.assertRaises(UnresolvableTimelineError):
            self._ingest([
                r(127, "2024-11-03T01:20:00", 80),
                r(129, "2024-11-03T01:20:00", 84),
            ])
        self.assertEqual(self._record_count(), 5)
        self.assertEqual(self._audit_statuses(), ["accepted", "rejected"])

    def test_empty_batch_rejected(self):
        with self.assertRaises(IngestionError):
            self._ingest([])

    def test_csv_parsing_rules(self):
        good = (
            "sample_id,local_time,pass_count\n"
            "128,2024-11-03T01:20:00,340\n"
            "148,2024-11-03T01:55:00,300\n"
        ).encode()
        records = parse_counter_csv(good)
        self.assertEqual(len(records), 2)

        with self.assertRaises(CsvValidationError):
            parse_counter_csv(b"id,local_time,pass_count\n1,x,2\n")
        with self.assertRaises(CsvValidationError):
            parse_counter_csv(
                b"sample_id,local_time,pass_count\n"
                b"1,2024-11-03T01:20:00-04:00,2\n"
            )
        with self.assertRaises(CsvValidationError):
            parse_counter_csv(
                b"sample_id,local_time,pass_count\n"
                b"1,2024-11-03T01:20:00,2\n1,2024-11-03T01:20:00,2\n"
            )
        with self.assertRaises(CsvValidationError):
            parse_counter_csv(b"sample_id,local_time,pass_count\n")
        with self.assertRaises(CsvValidationError):
            parse_counter_csv(
                b"sample_id,local_time,pass_count\n0,2024-11-03T01:20:00,2\n"
            )
        with self.assertRaises(CsvValidationError):
            parse_counter_csv(
                b"sample_id,local_time,pass_count\n1,2024-11-03T01:20:00,-2\n"
            )

    # ------------------------------------------------------------------
    # Decision/audit atomicity (the regression this suite guards)
    # ------------------------------------------------------------------

    def _block_audit_writes(self) -> None:
        """Make every insert into batch_audit fail (simulated audit outage)."""
        self.conn.execute(
            "CREATE TRIGGER fail_audit_insert BEFORE INSERT ON batch_audit "
            "BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END"
        )

    def test_acceptance_audit_failure_rolls_back_new_readings(self):
        self._block_audit_writes()
        with self.assertRaises(AuditPersistError):
            self._ingest([r(128, "2024-11-03T01:20:00", 340)])
        # no un-audited readings, no half-written audit row
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM batch_audit").fetchone()[0],
            0,
        )

    def test_idempotent_reupload_also_requires_its_audit_row(self):
        # first acceptance works and is audited
        self._ingest([r(128, "2024-11-03T01:20:00", 340)])
        self._block_audit_writes()
        # the same-value re-upload cannot be acknowledged without a new audit
        with self.assertRaises(AuditPersistError):
            self._ingest([r(128, "2024-11-03T01:20:00", 340)])
        self.assertEqual(self._record_count(), 1)
        self.assertEqual(self._audit_statuses(), ["accepted"])

    def test_rejection_audit_failure_is_an_error_not_a_recorded_reject(self):
        self._ingest([r(128, "2024-11-03T01:20:00", 340)])
        self._block_audit_writes()
        with self.assertRaises(AuditPersistError):
            self._ingest([r(128, "2024-11-03T01:20:00", 999)])
        # the conflicting value never landed and no phantom rejection exists
        self.assertEqual(
            self.conn.execute(
                "SELECT pass_count FROM records WHERE sample_id = 128"
            ).fetchone()[0],
            340,
        )
        self.assertEqual(self._audit_statuses(), ["accepted"])

    def test_rejected_batch_persists_reason_without_any_new_reading(self):
        self._ingest([r(100, "2024-11-03T00:45:00", 120)])
        with self.assertRaises(UnresolvableTimelineError):
            self._ingest([
                r(5, "2024-11-03T03:00:00", 50),
                r(15, "2024-11-03T00:30:00", 60),
            ])
        self.assertEqual(self._record_count(), 1)
        rows = list_audit(self.conn)
        self.assertEqual([row["status"] for row in rows],
                         ["accepted", "rejected"])
        self.assertIn("no legal timeline", rows[1]["reason"])

    def test_interleaved_decisions_are_listed_in_effectiveness_order(self):
        self._ingest([r(100, "2024-11-03T00:45:00", 120)])          # accepted
        with self.assertRaises(ConflictError):
            self._ingest([r(100, "2024-11-03T00:45:00", 999)])      # rejected
        self._ingest([r(160, "2024-11-03T02:15:00", 260)])          # accepted
        with self.assertRaises(UnresolvableTimelineError):
            self._ingest([
                r(5, "2024-11-03T03:00:00", 50),
                r(15, "2024-11-03T00:30:00", 60),
            ])                                                        # rejected
        self._ingest([r(180, "2024-11-03T03:10:00", 210)])          # accepted

        rows = list_audit(self.conn)
        self.assertEqual(
            [row["status"] for row in rows],
            ["accepted", "rejected", "accepted", "rejected", "accepted"],
        )
        # autoincrement ids are gapless and ordered exactly like the list
        self.assertEqual([row["id"] for row in rows], [1, 2, 3, 4, 5])
        # rejection reasons travel with their rows
        self.assertIn("sample_id 100", rows[1]["reason"])
        self.assertIn("no legal timeline", rows[3]["reason"])


if __name__ == "__main__":
    unittest.main()
