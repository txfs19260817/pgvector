-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.0'" to load this file. \quit

CREATE FUNCTION vector_dims(sparsevec) RETURNS integer
    AS 'MODULE_PATHNAME', 'sparsevec_vector_dims' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION vector_dims(sparsevec) IS 'number of dimensions';

CREATE FUNCTION bit_vector_negative_inner_product(bit, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION bit_vector_negative_inner_product(bit, vector) IS 'negative inner product of a sign code and vector';

CREATE FUNCTION vector_bit_negative_inner_product(vector, bit) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION vector_bit_negative_inner_product(vector, bit) IS 'negative inner product of a sign code and vector';

CREATE OPERATOR <#> (
    LEFTARG = bit, RIGHTARG = vector, PROCEDURE = bit_vector_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <#> (
    LEFTARG = vector, RIGHTARG = bit, PROCEDURE = vector_bit_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE FUNCTION hnsw_bit_vector_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE OPERATOR CLASS bit_vector_ip_ops
    FOR TYPE bit USING hnsw AS
    OPERATOR 1 <#> (bit, vector) FOR ORDER BY float_ops,
    FUNCTION 1 hamming_distance(bit, bit),
    FUNCTION 3 hnsw_bit_vector_support(internal),
    FUNCTION 4 vector_bit_negative_inner_product(vector, bit);
