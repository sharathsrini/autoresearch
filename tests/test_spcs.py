"""
Tests for the SPCS job runner against a fake Snowflake connection (no account, no docker).

    python tests/test_spcs.py
"""

import os
import re
import sys
import json
import shutil
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["SHAPE_AE_CACHE"] = tempfile.mkdtemp(prefix="shape_ae_spcs_")

import data_sources as ds  # noqa: E402
import spcs_run  # noqa: E402

spcs_run.POLL_SECONDS = 0
spcs_run.RUNS_DIR = os.path.join(os.environ["SHAPE_AE_CACHE"], "runs")

class FakeCursor:
    def __init__(self, conn):
        self.conn, self.rows, self.description = conn, [], [("X",)]

    def execute(self, sql, params=None):
        c = self.conn
        c.sql.append(sql)
        if sql.startswith("SHOW IMAGES"):
            self.description = [("IMAGE_NAME",), ("TAGS",)]
            self.rows = [("shape-ae", json.dumps([t])) for t in c.tags]
        elif sql.startswith("EXECUTE JOB SERVICE"):
            c.spec = json.loads(re.search(r"\$\$(.*)\$\$", sql, re.S).group(1))
            c.polls = 0
        elif "GET_SERVICE_STATUS" in sql:
            c.polls += 1
            st = c.final_status if c.polls >= 2 else "RUNNING"
            self.rows = [(json.dumps([{"status": st, "containerName": "main"}]),)]
        elif "GET_SERVICE_LOGS" in sql:
            self.rows = [("container log line",)]
        elif sql.startswith("GET "):
            dest = re.search(r"'file://([^']+)'", sql).group(1)
            if "outputs/" in sql:
                open(os.path.join(dest, "TTF_scores.csv"), "w").write("TRADE_DATE\n")
            else:
                open(os.path.join(dest, "run.log"), "w").write(c.run_log)
                open(os.path.join(dest, "exit_code"), "w").write(c.exit_code)
        self.rows = self.rows or []

    def fetchall(self):
        return self.rows

    def fetch_pandas_all(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=[d[0] for d in self.description])

    def close(self):
        pass

class FakeConn:
    def __init__(self, tags=(), final_status="DONE", run_log="val_score:        0.5\n", exit_code="0"):
        self.sql, self.tags, self.final_status = [], list(tags), final_status
        self.run_log, self.exit_code, self.spec, self.polls = run_log, exit_code, None, 0

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass

def use(conn):
    spcs_run.ds.connect = lambda cfg: conn

def cfg():
    return ds.load_config()

def test_tag_tracks_harness_but_not_train_py():
    t1 = spcs_run.harness_tag()
    assert re.fullmatch(r"h[0-9a-f]{12}", t1)
    assert "train.py" not in spcs_run.harness_files() and "prepare.py" in spcs_run.harness_files()
    assert any(f.startswith("tests/") for f in spcs_run.harness_files())

def test_spec_shape():
    spec = spcs_run.job_spec(cfg(), "habc", ["train", "r1"], {"JOB_TIMEOUT_SECONDS": 900})
    c = spec["spec"]["containers"][0]
    assert c["image"] == "/fera_db/shape_ae/shape_ae_images/shape-ae:habc"
    assert c["args"] == ["train", "r1"] and c["env"]["JOB_TIMEOUT_SECONDS"] == "900"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert spec["spec"]["volumes"][0]["source"] == "@FERA_DB.SHAPE_AE.SHAPE_AE_STAGE"
    assert c["volumeMounts"][0]["mountPath"] == "/stage"

def test_train_refuses_without_image(capsys=None):
    conn = FakeConn(tags=["hdeadbeef0000"])
    use(conn)
    try:
        spcs_run.cmd_train(cfg())
    except SystemExit as e:
        assert "spcs_run.py build" in str(e)
        assert not any(s.startswith("EXECUTE") for s in conn.sql)
        return
    raise AssertionError("ran without the image")

def test_train_happy_path():
    conn = FakeConn(tags=[spcs_run.harness_tag()])
    use(conn)
    spcs_run.cmd_train(cfg())
    kinds = [s.split()[0] for s in conn.sql]
    assert kinds.index("PUT") < kinds.index("EXECUTE") < kinds.index("GET") and "DROP" in kinds
    assert conn.spec["spec"]["containers"][0]["args"][0] == "train"
    run_dirs = os.listdir(spcs_run.RUNS_DIR)
    d = os.path.join(spcs_run.RUNS_DIR, sorted(run_dirs)[-1])
    assert open(os.path.join(d, "run.log")).read().startswith("val_score")
    assert os.path.exists(os.path.join(d, "outputs", "TTF_scores.csv"))
    assert os.path.exists(os.path.join(d, "train.py"))

def test_train_failed_exit_code_fails():
    conn = FakeConn(tags=[spcs_run.harness_tag()], exit_code="1", run_log="Traceback ...\n")
    use(conn)
    try:
        spcs_run.cmd_train(cfg())
    except SystemExit as e:
        assert e.code == 1
        assert "DROP" in [s.split()[0] for s in conn.sql]
        return
    raise AssertionError("failed run reported as success")

def test_timeout_drops_service():
    conn = FakeConn(tags=[spcs_run.harness_tag()], final_status="RUNNING")
    use(conn)
    c = cfg()
    c["spcs"]["job_timeout_minutes"] = 0
    real = spcs_run.execute_job

    def fast(conn_, cfg_, name, spec, timeout_s):
        return real(conn_, cfg_, name, spec, 0)
    spcs_run.execute_job = fast
    try:
        spcs_run.cmd_train(c)
    except SystemExit as e:
        assert e.code == 1 and any(s.startswith("DROP SERVICE") for s in conn.sql)
        return
    finally:
        spcs_run.execute_job = real
    raise AssertionError("timeout not reported")

def test_service_names_are_identifiers():
    n = spcs_run.service_name(cfg(), "SHAPE_AE_RUN", "abc1234-20261020101010-0f0f0f")
    assert ds._ident(n) == n and n.startswith("FERA_DB.SHAPE_AE.SHAPE_AE_RUN_ABC1234_")

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}")
    shutil.rmtree(spcs_run.RUNS_DIR, ignore_errors=True)
    sys.exit(1 if fails else 0)
