CREATE TABLE IF NOT EXISTS word (
  label TEXT PRIMARY KEY,
  ko_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS translate_log (
  id BIGSERIAL PRIMARY KEY,
  label TEXT,
  ko_text TEXT,
  confidence DOUBLE PRECISION,
  created_at TIMESTAMP DEFAULT now()
);
