"""HTTP-level tests via FastAPI's TestClient.

These require the web stack (fastapi, python-multipart, httpx). They run in
the container image / CI where requirements.txt is installed; in a stdlib-only
environment the whole module is skipped so the core suite stays runnable.
"""

import tempfile
import unittest
from pathlib import Path

import app.db as db_module


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"web stack unavailable: {exc}")

        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_file = Path(cls.tmp.name) / "api.db"
        cls._orig = db_module.DB_PATH
        db_module.DB_PATH = cls.db_file

        from app.main import app

        # lifespan opens the bootstrap connection against the patched path;
        # request-scoped connections are created per request.
        cls.client = TestClient(app)
        cls.client.__enter__()

    def setUp(self):
        # Every HTTP test starts from an empty warehouse while keeping the
        # schema (and any triggers) intact.
        admin = db_module.connect(self.db_file)
        try:
            admin.execute("DELETE FROM records")
            admin.execute("DELETE FROM batch_audit")
            admin.execute(
                "DELETE FROM sqlite_sequence WHERE name = 'batch_audit'"
            )
        finally:
            admin.close()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "client"):
            cls.client.__exit__(None, None, None)
        db_module.DB_PATH = cls._orig
        cls.tmp.cleanup()

    def _upload(self, name: str, content: bytes):
        return self.client.post(
            "/upload",
            files={"file": (name, content, "text/csv")},
        )

    def test_dst_flow_over_http(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

        initial = (
            "sample_id,local_time,pass_count\n"
            "100,2024-11-03T00:45:00,120\n"
            "128,2024-11-03T01:20:00,340\n"
            "148,2024-11-03T01:55:00,300\n"
            "160,2024-11-03T02:15:00,260\n"
            "180,2024-11-03T03:10:00,210\n"
        ).encode()
        r = self._upload("initial.csv", initial)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(sorted(r.json()["inserted_sample_ids"]),
                         [100, 128, 148, 160, 180])

        r = self.client.get("/traffic").json()
        self.assertEqual(r["pending_sample_ids"], [128, 148])
        self.assertEqual(r["determined_count"], 3)

        # idempotent re-upload: still an accepted decision with its own audit
        r = self._upload("initial.csv", initial)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["inserted_sample_ids"], [])
        self.assertEqual(
            sorted(r.json()["skipped_duplicate_sample_ids"]),
            [100, 128, 148, 160, 180],
        )

        # conflict -> 409 with a durable rejection reason
        r = self._upload(
            "conflict.csv",
            b"sample_id,local_time,pass_count\n128,2024-11-03T01:20:00,999\n",
        )
        self.assertEqual(r.status_code, 409)
        self.assertFalse(r.json()["accepted"])
        self.assertIn("sample_id 128", r.json()["error"])

        # illegal timeline -> 422
        r = self._upload(
            "illegal.csv",
            b"sample_id,local_time,pass_count\n"
            b"127,2024-11-03T01:20:00,80\n129,2024-11-03T01:20:00,84\n",
        )
        self.assertEqual(r.status_code, 422)

        # disambiguating backfill
        r = self._upload(
            "backfill.csv",
            b"sample_id,local_time,pass_count\n127,2024-11-03T01:20:00,84\n",
        )
        self.assertEqual(r.status_code, 200)

        r = self.client.get("/traffic").json()
        self.assertEqual(r["pending_sample_ids"], [])
        self.assertEqual(r["determined_count"], 6)
        by_id = {d["sample_id"]: d["utc_time"] for d in r["details"]}
        self.assertEqual(by_id[127], "2024-11-03T05:20:00+00:00")
        self.assertEqual(by_id[128], "2024-11-03T06:20:00+00:00")
        self.assertEqual(by_id[148], "2024-11-03T06:55:00+00:00")

        audits = self.client.get("/batches").json()["batches"]
        self.assertEqual(
            [a["status"] for a in audits],
            ["accepted", "accepted", "rejected", "rejected", "accepted"],
        )
        # ids expose the decision-effectiveness ordering
        self.assertEqual([a["id"] for a in audits], [1, 2, 3, 4, 5])
        self.assertIn("sample_id 128", audits[2]["reason"])
        self.assertIn("no legal timeline", audits[3]["reason"])

    def test_malformed_csv_returns_400(self):
        r = self._upload("bad.csv", b"id,local_time,pass_count\n1,x,2\n")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.json()["accepted"])

    def test_acceptance_audit_outage_returns_500_and_leaves_nothing(self):
        # Simulate an audit-write failure at the storage layer.
        admin = db_module.connect(self.db_file)
        admin.execute(
            "CREATE TRIGGER fail_audit_insert BEFORE INSERT ON batch_audit "
            "BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END"
        )
        admin.close()
        try:
            r = self._upload(
                "will-fail.csv",
                b"sample_id,local_time,pass_count\n300,2024-11-03T05:00:00,7\n",
            )
            self.assertEqual(r.status_code, 500)
            body = r.json()
            self.assertFalse(body["accepted"])
            self.assertIn("audit", body["error"])
        finally:
            admin = db_module.connect(self.db_file)
            admin.execute("DROP TRIGGER fail_audit_insert")
            admin.close()

        # the un-audited reading never landed, and no audit row was created
        traffic = self.client.get("/traffic").json()
        self.assertNotIn(300, [d["sample_id"] for d in traffic["details"]])
        audits = self.client.get("/batches").json()["batches"]
        self.assertFalse(
            any("300" in (a.get("reason") or "") for a in audits)
        )

        # after the outage clears, normal acceptance works and is audited
        r = self._upload(
            "recovered.csv",
            b"sample_id,local_time,pass_count\n300,2024-11-03T05:00:00,7\n",
        )
        self.assertEqual(r.status_code, 200, r.text)
        audits = self.client.get("/batches").json()["batches"]
        self.assertEqual(audits[-1]["status"], "accepted")

    def test_rejection_audit_outage_is_not_reported_as_recorded(self):
        # seed one reading
        r = self._upload(
            "seed.csv",
            b"sample_id,local_time,pass_count\n400,2024-11-03T06:00:00,9\n",
        )
        self.assertEqual(r.status_code, 200)

        admin = db_module.connect(self.db_file)
        admin.execute(
            "CREATE TRIGGER fail_audit_insert BEFORE INSERT ON batch_audit "
            "BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END"
        )
        admin.close()
        try:
            r = self._upload(
                "conflict-no-audit.csv",
                b"sample_id,local_time,pass_count\n400,2024-11-03T06:00:00,11\n",
            )
            # must not be a 409 pretending the rejection was recorded
            self.assertEqual(r.status_code, 500)
            self.assertIn("rejected audit", r.json()["error"])
        finally:
            admin = db_module.connect(self.db_file)
            admin.execute("DROP TRIGGER fail_audit_insert")
            admin.close()

        # original value intact, and only the seed acceptance is on record
        traffic = self.client.get("/traffic").json()
        by_id = {d["sample_id"]: d["pass_count"] for d in traffic["details"]}
        self.assertEqual(by_id[400], 9)
        audits = self.client.get("/batches").json()["batches"]
        statuses = [a["status"] for a in audits]
        self.assertEqual(statuses.count("rejected"), 0)


if __name__ == "__main__":
    unittest.main()
