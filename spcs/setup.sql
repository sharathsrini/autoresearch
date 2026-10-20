-- One-time setup for the Shape AE SPCS job runner. Names match the [spcs] section of fera.toml;
-- change both together. Run as a role allowed to create compute pools (e.g. ACCOUNTADMIN or a
-- delegated role), then grant usage to the role that runs spcs_run.py.

USE DATABASE FERA_DB;
CREATE SCHEMA IF NOT EXISTS SHAPE_AE;
USE SCHEMA SHAPE_AE;

-- GPU pool: one small NVIDIA node is plenty (models are ~50k parameters). Suspends when idle.
CREATE COMPUTE POOL IF NOT EXISTS SHAPE_AE_GPU_POOL
  MIN_NODES = 1 MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_SUSPEND_SECS = 1800
  AUTO_RESUME = TRUE;

CREATE IMAGE REPOSITORY IF NOT EXISTS SHAPE_AE_IMAGES;
SHOW IMAGE REPOSITORIES LIKE 'SHAPE_AE_IMAGES';  -- repository_url is the docker push target

-- Stage mounted at /stage in every job. Server-side encryption is required for stage mounts.
CREATE STAGE IF NOT EXISTS SHAPE_AE_STAGE
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE);

-- Role that runs experiments (replace FERA_RESEARCH with yours)
GRANT USAGE, MONITOR ON COMPUTE POOL SHAPE_AE_GPU_POOL TO ROLE FERA_RESEARCH;
GRANT READ, WRITE ON IMAGE REPOSITORY SHAPE_AE_IMAGES TO ROLE FERA_RESEARCH;
GRANT READ, WRITE ON STAGE SHAPE_AE_STAGE TO ROLE FERA_RESEARCH;
GRANT CREATE SERVICE ON SCHEMA SHAPE_AE TO ROLE FERA_RESEARCH;
-- The prepare job reads the curve view (and the feature store, if configured) as this role:
GRANT USAGE ON DATABASE <CURVE_DB> TO ROLE FERA_RESEARCH;
GRANT USAGE ON SCHEMA <CURVE_DB>.RW_CF_FERA TO ROLE FERA_RESEARCH;
GRANT SELECT ON VIEW <CURVE_DB>.RW_CF_FERA.VW_ML_LONG TO ROLE FERA_RESEARCH;
