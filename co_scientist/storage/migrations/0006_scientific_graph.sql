-- Structured scientific state.  Rows are append-friendly so each controller
-- decision can be audited against the graph it observed.
CREATE TABLE IF NOT EXISTS scientific_nodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'unknown',
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(session_id, id)
);
CREATE INDEX IF NOT EXISTS sci_nodes_session ON scientific_nodes(session_id, kind);

CREATE TABLE IF NOT EXISTS scientific_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0,
    unresolved INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, source, target, relation)
);
CREATE INDEX IF NOT EXISTS sci_edges_session ON scientific_edges(session_id, unresolved);

CREATE TABLE IF NOT EXISTS control_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    target_edge TEXT,
    expected_information_gain REAL NOT NULL DEFAULT 0,
    graph_coverage REAL NOT NULL DEFAULT 0,
    budget_remaining REAL,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS control_session ON control_decisions(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS experiment_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES tasks(id),
    title TEXT NOT NULL,
    candidates TEXT NOT NULL DEFAULT '[]',
    distinction TEXT NOT NULL DEFAULT '',
    intervention TEXT NOT NULL DEFAULT '',
    positive_prediction TEXT NOT NULL DEFAULT '',
    negative_prediction TEXT NOT NULL DEFAULT '',
    falsification_rule TEXT NOT NULL DEFAULT '',
    information_gain REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'proposed'
);
CREATE INDEX IF NOT EXISTS experiments_session ON experiment_proposals(session_id, status);
