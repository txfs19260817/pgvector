SET enable_seqscan = off;

-- hamming

CREATE TABLE t (val bit(3));
INSERT INTO t (val) VALUES (B'000'), (B'100'), (B'111'), (NULL);
CREATE INDEX ON t USING hnsw (val bit_hamming_ops);

INSERT INTO t (val) VALUES (B'110');

SELECT * FROM t ORDER BY val <~> B'111';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <~> (SELECT NULL::bit)) t2;

DROP TABLE t;

-- asymmetric bit query

CREATE TABLE t (id int, val vector(2));
INSERT INTO t VALUES (1, '[1,-1]'), (2, '[-1,1]'), (3, '[1,1]');
CREATE INDEX bit_idx ON t USING hnsw ((binary_quantize(val)::bit(2)) bit_vector_ip_ops);
INSERT INTO t VALUES (4, '[-1,-1]');

SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> '[2,3]'::vector;
EXPLAIN (COSTS OFF) SELECT id FROM t ORDER BY '[2,3]'::vector <#> binary_quantize(val)::bit(2) LIMIT 1;
SELECT id FROM t ORDER BY '[2,3]'::vector <#> binary_quantize(val)::bit(2);
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY binary_quantize(val)::bit(2) <#> (SELECT NULL::vector)) t2;
SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> '[1,2,3]'::vector LIMIT 1;
SELECT q, ARRAY(SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> q LIMIT 4)
FROM (VALUES ('[2,3]'::vector), ('[-2,-3]'::vector)) v(q);
DELETE FROM t WHERE id = 2;
VACUUM t;
SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> '[2,3]'::vector;
DROP TABLE t;

-- asymmetric bit query on halfvec storage

CREATE TABLE t (id int, val halfvec(2));
INSERT INTO t VALUES (1, '[1,-1]'), (2, '[-1,1]'), (3, '[1,1]'), (4, '[-1,-1]');
CREATE INDEX bit_idx ON t USING hnsw ((binary_quantize(val)::bit(2)) bit_vector_ip_ops);
EXPLAIN (COSTS OFF) SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> '[2,3]'::vector LIMIT 1;
SELECT id FROM t ORDER BY binary_quantize(val)::bit(2) <#> '[2,3]'::vector;
DROP TABLE t;

-- codes cannot have more dimensions than a vector query

CREATE TABLE t (val bit(16001));
CREATE INDEX ON t USING hnsw (val bit_vector_ip_ops);
DROP TABLE t;

-- query type without a query distance function

CREATE OPERATOR CLASS bit_vector_invalid_ops FOR TYPE bit USING hnsw AS
	OPERATOR 1 <#> (bit, vector) FOR ORDER BY float_ops,
	FUNCTION 1 hamming_distance(bit, bit);
SELECT amvalidate(oid) FROM pg_opclass WHERE opcname = 'bit_vector_invalid_ops';
DROP OPERATOR FAMILY bit_vector_invalid_ops USING hnsw;

-- jaccard

CREATE TABLE t (val bit(4));
INSERT INTO t (val) VALUES (B'0000'), (B'1100'), (B'1111'), (NULL);
CREATE INDEX ON t USING hnsw (val bit_jaccard_ops);

INSERT INTO t (val) VALUES (B'1110');

SELECT * FROM t ORDER BY val <%> B'1111';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <%> (SELECT NULL::bit)) t2;

DROP TABLE t;

-- varbit

CREATE TABLE t (val varbit(3));
CREATE INDEX ON t USING hnsw (val bit_hamming_ops);
CREATE INDEX ON t USING hnsw ((val::bit(3)) bit_hamming_ops);
CREATE INDEX ON t USING hnsw ((val::bit(64001)) bit_hamming_ops);
DROP TABLE t;

-- dimensions

CREATE TABLE t (val bit(64000));
CREATE INDEX ON t USING hnsw (val bit_hamming_ops);
DROP TABLE t;

CREATE TABLE t (val bit(64001));
CREATE INDEX ON t USING hnsw (val bit_hamming_ops);
DROP TABLE t;
