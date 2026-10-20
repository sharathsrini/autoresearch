"""
Run Shape AE autoresearch jobs on Snowpark Container Services (fixed, do not modify from the agent loop).

    uv run spcs_run.py build                 # build + push the harness image (tag = harness hash)
    uv run spcs_run.py prepare [--markets TTF,THE]   # pull from Snowflake inside SPCS -> /stage/cache
    uv run spcs_run.py train > run.log 2>&1  # one experiment: upload train.py, run on the GPU pool, fetch logs
    uv run spcs_run.py train --local > run.log 2>&1  # same interface, runs on this machine
    uv run spcs_run.py tests                 # run the harness tests inside the image

Each experiment gets a run id <commit>-<utc time>-<train.py hash>; its train.py, run.log, exit code and
outputs live under @<stage>/runs/<run id>/ and are copied to ./runs/<run id>/. The run log is printed to
stdout, so `grep "^val_score:" run.log` works the same as a local run.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess
import datetime as dt

import data_sources as ds

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(REPO, "runs")
HARNESS_FILES = ["prepare.py", "data_sources.py", "events.py", "fera.toml",
                 "spcs/Dockerfile", "spcs/entrypoint.sh"]
POLL_SECONDS = 10

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def harness_files():
    tests = sorted(os.path.join("tests", f) for f in os.listdir(os.path.join(REPO, "tests")) if f.endswith(".py"))
    return HARNESS_FILES + tests

def harness_tag():
    """Image tag = hash of every file baked into the image. Any harness change needs a new image."""
    h = hashlib.sha256()
    for f in harness_files():
        h.update(f.encode())
        with open(os.path.join(REPO, f), "rb") as fh:
            h.update(fh.read())
    return "h" + h.hexdigest()[:12]

def run_id(train_path):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "nogit"
    with open(train_path, "rb") as f:
        th = hashlib.sha256(f.read()).hexdigest()[:6]
    return f"{commit}-{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}-{th}".lower()

def service_name(cfg, prefix, rid):
    s = cfg["spcs"]
    return f"{s['database']}.{s['schema']}.{prefix}_{rid.replace('-', '_')}".upper()

# ---------------------------------------------------------------------------
# Snowflake helpers
# ---------------------------------------------------------------------------

def q(conn, sql, params=None):
    cur = conn.cursor()
    try:
        cur.execute(sql, params or {})
        return cur.fetchall()
    finally:
        cur.close()

def stage_ref(cfg, path=""):
    s = cfg["spcs"]
    return f"@{ds._ident(s['database'])}.{ds._ident(s['schema'])}.{ds._ident(s['stage'])}/{path}"

def image_path(cfg, tag):
    s = cfg["spcs"]
    return f"/{s['database']}/{s['schema']}/{s['image_repository']}/{s['image_name']}:{tag}".lower()

def image_exists(conn, cfg, tag):
    s = cfg["spcs"]
    repo = f"{ds._ident(s['database'])}.{ds._ident(s['schema'])}.{ds._ident(s['image_repository'])}"
    df = ds._query_df(conn, f"SHOW IMAGES IN IMAGE REPOSITORY {repo}", {})
    if df.empty:
        return False
    names = df.get("IMAGE_NAME", df.iloc[:, 0]).astype(str).str.lower()
    tags = df.get("TAGS", df.get("TAG", "")).astype(str)
    return bool(((names.str.endswith(s["image_name"].lower())) & tags.str.contains(tag)).any())

def job_spec(cfg, tag, args, env=None):
    """SPCS job specification. JSON is valid YAML, so json.dumps is the renderer."""
    s = cfg["spcs"]
    container = {"name": "main", "image": image_path(cfg, tag), "args": list(args),
                 "env": {k: str(v) for k, v in (env or {}).items()},
                 "volumeMounts": [{"name": "stage", "mountPath": "/stage"}]}
    if int(s.get("gpus", 0)) > 0:
        g = str(int(s["gpus"]))
        container["resources"] = {"requests": {"nvidia.com/gpu": g}, "limits": {"nvidia.com/gpu": g}}
    return {"spec": {"containers": [container],
                     "volumes": [{"name": "stage", "source": stage_ref(cfg).rstrip("/")}]}}

def execute_job(conn, cfg, name, spec, timeout_s):
    """Start a job service asynchronously, poll until it ends or times out. Returns (status, logs)."""
    s = cfg["spcs"]
    wh = f" QUERY_WAREHOUSE = {ds._ident(s['query_warehouse'])}" if s.get("query_warehouse") else ""
    q(conn, f"EXECUTE JOB SERVICE IN COMPUTE POOL {ds._ident(s['compute_pool'])} NAME = {ds._ident(name)}"
            f"{wh} ASYNC = TRUE FROM SPECIFICATION $$\n{json.dumps(spec, indent=2)}\n$$")
    t0, status = time.time(), "PENDING"
    try:
        while True:
            raw = q(conn, "SELECT SYSTEM$GET_SERVICE_STATUS(%(n)s)", {"n": name})[0][0]
            states = [c.get("status", "") for c in json.loads(raw)] if raw else []
            status = states[0] if states else "UNKNOWN"
            if status in ("DONE", "FAILED", "INTERNAL_ERROR"):
                break
            if time.time() - t0 > timeout_s:
                status = "TIMEOUT"
                break
            time.sleep(POLL_SECONDS)
        try:
            logs = q(conn, "SELECT SYSTEM$GET_SERVICE_LOGS(%(n)s, 0, 'main', 1000)", {"n": name})[0][0]
        except Exception as e:  # logs are best-effort
            logs = f"(no service logs: {e})"
    finally:
        q(conn, f"DROP SERVICE IF EXISTS {ds._ident(name)}")
    return status, logs

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build(cfg, dry_run=False):
    s, tag = cfg["spcs"], harness_tag()
    conn = ds.connect(cfg)
    try:
        df = ds._query_df(conn, f"SHOW IMAGE REPOSITORIES LIKE %(r)s IN SCHEMA "
                                f"{ds._ident(s['database'])}.{ds._ident(s['schema'])}", {"r": s["image_repository"]})
    finally:
        conn.close()
    if df.empty:
        raise SystemExit(f"image repository {s['image_repository']} not found: run spcs/setup.sql first")
    url = df["REPOSITORY_URL"].iloc[0]
    image = f"{url}/{s['image_name']}:{tag}"
    cmds = [["docker", "build", "--platform", "linux/amd64", "-f", "spcs/Dockerfile", "-t", image, "."],
            ["docker", "push", image]]
    print(f"harness tag {tag}\nlog in first if needed: snow spcs image-registry login  (or: docker login {url.split('/')[0]})")
    for c in cmds:
        print("$ " + " ".join(c))
        if not dry_run:
            if shutil.which("docker") is None:
                raise SystemExit("docker not found; run the commands above where docker is available")
            subprocess.run(c, cwd=REPO, check=True)

def _require_image(conn, cfg, tag):
    if not image_exists(conn, cfg, tag):
        raise SystemExit(f"image {cfg['spcs']['image_name']}:{tag} is not in the repository. The harness changed "
                         f"or was never pushed: run `uv run spcs_run.py build` first.")

def cmd_prepare(cfg, extra_args):
    tag = harness_tag()
    rid = run_id(os.path.join(REPO, "prepare.py"))
    name = service_name(cfg, "SHAPE_AE_PREP", rid)
    conn = ds.connect(cfg)
    try:
        _require_image(conn, cfg, tag)
        env = {"SHAPE_AE_WAREHOUSE": cfg["spcs"].get("query_warehouse", "")}
        status, logs = execute_job(conn, cfg, name, job_spec(cfg, tag, ["prepare"] + extra_args, env), 3600)
    finally:
        conn.close()
    print(logs)
    print(f"spcs: prepare job {name} {status}")
    if status != "DONE":
        sys.exit(1)

def cmd_tests(cfg):
    tag = harness_tag()
    name = service_name(cfg, "SHAPE_AE_TESTS", run_id(os.path.join(REPO, "prepare.py")))
    conn = ds.connect(cfg)
    try:
        _require_image(conn, cfg, tag)
        status, logs = execute_job(conn, cfg, name, job_spec(cfg, tag, ["tests"]), 1800)
    finally:
        conn.close()
    print(logs)
    print(f"spcs: tests job {status}")
    sys.exit(0 if status == "DONE" else 1)

def cmd_train(cfg, local=False):
    train = os.path.join(REPO, "train.py")
    rid = run_id(train)
    local_dir = os.path.join(RUNS_DIR, rid)
    os.makedirs(local_dir, exist_ok=True)
    shutil.copy(train, os.path.join(local_dir, "train.py"))
    timeout_s = int(cfg["spcs"]["job_timeout_minutes"]) * 60
    print(f"spcs: run {rid} ({'local' if local else 'SPCS'})", flush=True)

    if local:
        env = dict(os.environ, SHAPE_AE_OUTPUT_DIR=os.path.join(local_dir, "outputs"))
        try:
            p = subprocess.run([sys.executable, "train.py"], cwd=REPO, env=env, capture_output=True, text=True,
                               timeout=timeout_s)
            log, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            dec = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
            log, rc = dec(e.stdout) + dec(e.stderr) + f"\nkilled after {timeout_s}s\n", 124
        with open(os.path.join(local_dir, "run.log"), "w") as f:
            f.write(log)
        print(log)
        status = "DONE" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAILED")
    else:
        tag = harness_tag()
        name = service_name(cfg, "SHAPE_AE_RUN", rid)
        conn = ds.connect(cfg)
        try:
            _require_image(conn, cfg, tag)
            q(conn, f"PUT 'file://{train}' '{stage_ref(cfg, f'runs/{rid}/')}' AUTO_COMPRESS = FALSE OVERWRITE = TRUE")
            events = os.path.join(REPO, "events", "events.csv")
            if os.path.exists(events):
                q(conn, f"PUT 'file://{events}' '{stage_ref(cfg, f'runs/{rid}/')}' AUTO_COMPRESS = FALSE OVERWRITE = TRUE")
            spec = job_spec(cfg, tag, ["train", rid], {"JOB_TIMEOUT_SECONDS": timeout_s})
            status, service_logs = execute_job(conn, cfg, name, spec, timeout_s + 300)
            for sub, pattern in [("", r".*(run\.log|exit_code)"), ("outputs/", r".*")]:
                os.makedirs(os.path.join(local_dir, sub), exist_ok=True)
                try:
                    q(conn, f"GET '{stage_ref(cfg, f'runs/{rid}/{sub}')}' 'file://{os.path.join(local_dir, sub)}' "
                            f"PATTERN = '{pattern}'")
                except Exception as e:
                    print(f"spcs: could not fetch {sub or 'run files'}: {e}")
        finally:
            conn.close()
        with open(os.path.join(local_dir, "service.log"), "w") as f:
            f.write(service_logs or "")
        log_path = os.path.join(local_dir, "run.log")
        if os.path.exists(log_path):
            print(open(log_path).read())
        else:
            print(service_logs)
        code_path = os.path.join(local_dir, "exit_code")
        if os.path.exists(code_path) and open(code_path).read().strip() not in ("", "0"):
            status = "FAILED"
    print(f"spcs: run {rid} {status}; files in {local_dir}")
    if status != "DONE":
        print("FAIL")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shape AE jobs on Snowpark Container Services")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build and push the harness image")
    b.add_argument("--dry-run", action="store_true", help="print the docker commands only")
    p = sub.add_parser("prepare", help="run prepare.py --source snowflake inside SPCS")
    p.add_argument("--markets", default=None)
    t = sub.add_parser("train", help="run one experiment (train.py) and print its log")
    t.add_argument("--local", action="store_true", help="run on this machine instead of SPCS")
    sub.add_parser("tests", help="run the harness tests inside the image")
    sub.add_parser("tag", help="print the harness image tag")
    args = ap.parse_args()
    cfg = ds.load_config()
    if args.cmd == "build":
        cmd_build(cfg, args.dry_run)
    elif args.cmd == "prepare":
        cmd_prepare(cfg, ["--markets", args.markets] if args.markets else [])
    elif args.cmd == "train":
        cmd_train(cfg, args.local)
    elif args.cmd == "tests":
        cmd_tests(cfg)
    else:
        print(harness_tag())
