# Database operations

BESSPulse reads `DATABASE_URL`; its local fallback is `sqlite:///data/besspulse.db`. PostgreSQL and Supabase use the `postgresql+psycopg://` SQLAlchemy dialect. Engines enable connection pre-ping; PostgreSQL additionally uses bounded pooling, recycling, and a connection timeout. Logs report only the dialect and database name, never credentials.

Schema changes are Alembic-governed. Apply them with `alembic upgrade head` and verify metadata drift with `alembic check`. `Base.metadata.create_all()` remains only for isolated tests and legacy bootstrap compatibility, not production upgrades.

PostgreSQL integration tests are opt-in: set `RUN_POSTGRES_TESTS=1` and point `DATABASE_URL` at a database whose name contains `test` (or a localhost server). The guard rejects destructive tests against other remote targets. Supabase poolers may require the provider's pooler connection string and SSL settings.

Supabase REST/Auth keys and management access tokens are not application requirements in this phase. Alembic requires the Supabase PostgreSQL connection string in `DATABASE_URL`; a project URL or REST key cannot substitute for it.
