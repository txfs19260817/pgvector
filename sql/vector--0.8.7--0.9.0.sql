-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.0'" to load this file. \quit

CREATE FUNCTION vector_dims(sparsevec) RETURNS integer
    AS 'MODULE_PATHNAME', 'sparsevec_vector_dims' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION vector_dims(sparsevec) IS 'number of dimensions';

-- rabitq type

CREATE TYPE rabitq;

CREATE FUNCTION rabitq_in(cstring, oid, integer) RETURNS rabitq
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_out(rabitq) RETURNS cstring
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_recv(internal, oid, integer) RETURNS rabitq
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_send(rabitq) RETURNS bytea
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE rabitq (
    INPUT     = rabitq_in,
    OUTPUT    = rabitq_out,
    TYPMOD_IN = rabitq_typmod_in,
    RECEIVE   = rabitq_recv,
    SEND      = rabitq_send,
    ALIGNMENT = double,
    STORAGE   = external
);

COMMENT ON TYPE rabitq IS 'RaBitQ code';

-- rabitq functions

CREATE FUNCTION rabitq_cast(rabitq, integer, boolean) RETURNS rabitq
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (rabitq AS rabitq)
    WITH FUNCTION rabitq_cast(rabitq, integer, boolean) AS IMPLICIT;

CREATE FUNCTION rabitq_quantize(vector, integer DEFAULT 4) RETURNS rabitq
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_l2_distance(rabitq, rabitq) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_vector_l2_distance(rabitq, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_rabitq_l2_distance(vector, rabitq) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hnsw_rabitq_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION rabitq_query_l2_distance(internal, rabitq) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_prepare_query(vector, internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- rabitq operators

CREATE OPERATOR <-> (
    LEFTARG = rabitq, RIGHTARG = vector, PROCEDURE = rabitq_vector_l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <-> (
    LEFTARG = vector, RIGHTARG = rabitq, PROCEDURE = vector_rabitq_l2_distance,
    COMMUTATOR = '<->'
);

-- rabitq opclasses

CREATE OPERATOR CLASS rabitq_l2_ops
    FOR TYPE rabitq USING hnsw AS
    OPERATOR 1 <-> (rabitq, vector) FOR ORDER BY float_ops,
    FUNCTION 1 rabitq_l2_distance(rabitq, rabitq),
    FUNCTION 3 hnsw_rabitq_support(internal),
    FUNCTION 4 rabitq_query_l2_distance(internal, rabitq),
    FUNCTION 5 rabitq_prepare_query(vector, internal);
