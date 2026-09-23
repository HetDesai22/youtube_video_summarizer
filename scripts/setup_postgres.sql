-- Reference only: a from-scratch setup for the Chat with Video database.
-- Names/password must match DB_NAME / DB_USER / DB_PASSWORD in .env.
-- (The migration itself runs CREATE EXTENSION vector; the connecting role must
-- be a superuser, or run the CREATE EXTENSION line below once as one.)
--
-- Run as the postgres superuser (pgAdmin Query Tool or psql -U postgres):

CREATE ROLE videosummary_user WITH LOGIN PASSWORD 'CHOOSE_A_PASSWORD_HERE';
CREATE DATABASE videosummary_db OWNER videosummary_user;

-- Then connect to that database and enable pgvector:
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify:
-- SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
