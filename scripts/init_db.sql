-- Executed automatically by the pgvector/pgvector Docker image on first boot.
-- Enables required PostgreSQL extensions for ORIX.

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: face embedding search
CREATE EXTENSION IF NOT EXISTS pgcrypto;    -- gen_random_uuid() for IDs
