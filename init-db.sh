#!/bin/sh
set -e

# Create user and database
psql -v ON_ERROR_STOP=1 --username "postgres" <<-EOSQL
    CREATE USER langgraph_user WITH PASSWORD 'langgraph_password';
    CREATE DATABASE langgraph_db OWNER langgraph_user;
EOSQL

# Now connect to langgraph_db and grant privileges
psql -v ON_ERROR_STOP=1 --username "postgres" --dbname "langgraph_db" <<-EOSQL
    GRANT ALL ON SCHEMA public TO langgraph_user;
    GRANT CREATE ON SCHEMA public TO langgraph_user;
EOSQL
