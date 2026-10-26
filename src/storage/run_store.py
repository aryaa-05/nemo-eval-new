import json
import os
import sqlite3
from typing import List, Dict, Any, Optional
from src.storage.schema import RunArtifact

class RunStore:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)
        self.db_path = os.path.join(runs_dir, "runs_registry.db")
        self._init_db()
        self._backfill_run_modes()
        self._backfill_identity()

    def _backfill_run_modes(self):
        """Fill run_mode for rows written before the column existed, using the run file when possible."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, provider, artifact_path FROM runs WHERE run_mode IS NULL"
            ).fetchall()
            for run_id, provider, artifact_path in rows:
                mode = None
                for candidate in (artifact_path, os.path.join(self.runs_dir, f"{run_id}.json")):
                    if not candidate or not os.path.exists(candidate):
                        continue
                    try:
                        with open(candidate, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                        mode = data.get("run_mode")
                    except (OSError, ValueError):
                        mode = None
                    if mode:
                        break
                if not mode:
                    mode = "simulated" if provider == "mock" else "unknown"
                conn.execute("UPDATE runs SET run_mode = ? WHERE run_id = ?", (mode, run_id))
            conn.commit()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    job_name TEXT,
                    timestamp TEXT,
                    model_id TEXT,
                    model_name TEXT,
                    provider TEXT,
                    overall_accuracy REAL,
                    ci_lower REAL,
                    ci_upper REAL,
                    mean_latency REAL,
                    total_tokens INTEGER,
                    total_cost REAL,
                    artifact_path TEXT,
                    run_mode TEXT,
                    served_model_id TEXT,
                    model_substituted INTEGER,
                    route_note TEXT
                )
            """)
            # Older databases predate these columns; add them without losing existing rows.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            if "run_mode" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN run_mode TEXT")
            if "served_model_id" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN served_model_id TEXT")
            if "model_substituted" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN model_substituted INTEGER")
            if "route_note" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN route_note TEXT")
            conn.commit()

    def _backfill_identity(self):
        """Fill served_model_id / model_substituted / route_note for rows written before those columns existed."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, artifact_path, model_id, served_model_id, model_substituted, route_note FROM runs"
            ).fetchall()
            for run_id, artifact_path, model_id, served, substituted, note in rows:
                if served is not None and substituted is not None and note is not None:
                    continue
                data = None
                for candidate in (artifact_path, os.path.join(self.runs_dir, f"{run_id}.json")):
                    if not candidate or not os.path.exists(candidate):
                        continue
                    try:
                        with open(candidate, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                    except (OSError, ValueError):
                        data = None
                    if data is not None:
                        break
                if isinstance(data, dict):
                    served = data.get("served_model_id") or served or model_id
                    substituted = data.get("model_substituted")
                    substituted = 0 if substituted is None else int(bool(substituted))
                    note = data.get("route_note") or note or ""
                else:
                    served = served or model_id
                    substituted = 0 if substituted is None else int(substituted)
                    note = note or ""
                conn.execute(
                    "UPDATE runs SET served_model_id = ?, model_substituted = ?, route_note = ? WHERE run_id = ?",
                    (served, substituted, note, run_id),
                )
            conn.commit()

    def save_run(self, artifact: RunArtifact, register: bool = True) -> str:
        """Write the run file. A completed run is also entered in the run list / leaderboard unless register=False.
        A failed or incomplete run, and a re-scored copy, are kept as a JSON record only."""
        filepath = os.path.join(self.runs_dir, f"{artifact.run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2))

        if artifact.status != "completed" or not register:
            return filepath

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, job_name, timestamp, model_id, model_name, provider,
                    overall_accuracy, ci_lower, ci_upper, mean_latency, total_tokens, total_cost, artifact_path,
                    run_mode, served_model_id, model_substituted, route_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact.run_id,
                artifact.job_name,
                artifact.timestamp,
                artifact.model_id,
                artifact.model_name,
                artifact.provider,
                artifact.overall_accuracy,
                artifact.overall_ci_lower,
                artifact.overall_ci_upper,
                artifact.mean_latency,
                artifact.total_tokens,
                artifact.total_cost,
                filepath,
                artifact.run_mode,
                artifact.served_model_id or artifact.model_id,
                int(artifact.model_substituted),
                artifact.route_note or ""
            ))
            conn.commit()
        return filepath

    def save_checkpoint(self, artifact: RunArtifact) -> str:
        """Save a partial (INCOMPLETE) run after a finished benchmark. JSON only; the final save_run replaces it."""
        return self.save_run(artifact, register=False)

    def list_unusable(self) -> List[Dict[str, Any]]:
        """Run files that cannot be compared or gated: failed (stopped early) and incomplete (never finished)."""
        found = []
        for name in sorted(os.listdir(self.runs_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.runs_dir, name), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("status") in ("failed", "incomplete"):
                found.append({
                    "run_id": data.get("run_id", name[:-5]), "status": data["status"], "model_name": data.get("model_name", ""),
                    "job_name": data.get("job_name", ""), "timestamp": data.get("timestamp", ""),
                    "benchmarks_done": list((data.get("benchmark_summaries") or {}).keys()),
                    "planned": (data.get("execution_params") or {}).get("benchmarks", []),
                    "reason": data.get("failure_reason") or "",
                })
        return sorted(found, key=lambda r: r["timestamp"], reverse=True)

    def get_run(self, run_id: str) -> Optional[RunArtifact]:
        filepath = os.path.join(self.runs_dir, f"{run_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return RunArtifact.model_validate(data)

    def list_runs(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        # One row per (model, provider, mode). A simulated run and a live run of the same model are kept apart,
        # because mixing them would make the leaderboard misleading. The CI and latest job are taken from the
        # best (or newest) matching run instead of an arbitrary row in the group.
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT r.model_id, r.model_name, r.provider,
                       COALESCE(r.run_mode, 'unknown') as run_mode,
                       MAX(r.overall_accuracy) as best_accuracy,
                       (SELECT x.ci_lower FROM runs x
                         WHERE x.model_id = r.model_id
                           AND x.provider = r.provider
                           AND COALESCE(x.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY x.overall_accuracy DESC, x.run_id DESC LIMIT 1) as ci_lower,
                       (SELECT x.ci_upper FROM runs x
                         WHERE x.model_id = r.model_id
                           AND x.provider = r.provider
                           AND COALESCE(x.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY x.overall_accuracy DESC, x.run_id DESC LIMIT 1) as ci_upper,
                       AVG(r.mean_latency) as avg_latency,
                       SUM(r.total_tokens) as total_tokens,
                       SUM(r.total_cost) as total_cost,
                       COUNT(r.run_id) as total_runs,
                       (SELECT y.run_id FROM runs y
                         WHERE y.model_id = r.model_id
                           AND y.provider = r.provider
                           AND COALESCE(y.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY y.run_id DESC LIMIT 1) as latest_run_id,
                       (SELECT y.job_name FROM runs y
                         WHERE y.model_id = r.model_id
                           AND y.provider = r.provider
                           AND COALESCE(y.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY y.run_id DESC LIMIT 1) as latest_job_name,
                       COALESCE((SELECT y.served_model_id FROM runs y
                         WHERE y.model_id = r.model_id
                           AND y.provider = r.provider
                           AND COALESCE(y.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY y.run_id DESC LIMIT 1), r.model_id) as served_model_id,
                       COALESCE((SELECT y.model_substituted FROM runs y
                         WHERE y.model_id = r.model_id
                           AND y.provider = r.provider
                           AND COALESCE(y.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY y.run_id DESC LIMIT 1), 0) as model_substituted,
                       COALESCE((SELECT y.route_note FROM runs y
                         WHERE y.model_id = r.model_id
                           AND y.provider = r.provider
                           AND COALESCE(y.run_mode, 'unknown') = COALESCE(r.run_mode, 'unknown')
                         ORDER BY y.run_id DESC LIMIT 1), '') as route_note
                FROM runs r
                GROUP BY r.model_id, r.provider, COALESCE(r.run_mode, 'unknown')
                ORDER BY best_accuracy DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
