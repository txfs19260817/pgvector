SET enable_seqscan = off;

-- binary quantization on vector. Rows come back in approximate distance order, so the tests re-sort them
-- with a materialized CTE (rows with the same sign bits tie)

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[-1,-2,-3]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantization = 'binary');

INSERT INTO t (val) VALUES ('[1,2,4]');

EXPLAIN (COSTS OFF) SELECT * FROM t ORDER BY val <-> '[3,3,3]' LIMIT 5;
WITH candidates AS MATERIALIZED (SELECT val, val <-> '[3,3,3]' AS distance FROM t ORDER BY distance LIMIT 10)
SELECT val FROM candidates ORDER BY distance + 0;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> '[3,3,3]') t2;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::vector)) t2;

SET hnsw.iterative_scan = strict_order;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> '[3,3,3]') t2;
RESET hnsw.iterative_scan;

DELETE FROM t WHERE val = '[1,1,1]';
VACUUM t;
WITH candidates AS MATERIALIZED (SELECT val, val <-> '[3,3,3]' AS distance FROM t ORDER BY distance LIMIT 10)
SELECT val FROM candidates ORDER BY distance + 0;

DROP TABLE t;

-- cosine and inner product

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[-1,-2,-3]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_cosine_ops) WITH (quantization = 'binary');
CREATE INDEX ON t USING hnsw (val vector_ip_ops) WITH (quantization = 'binary');

SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> '[3,3,3]') t2;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <#> '[3,3,3]') t2;

DROP TABLE t;

-- halfvec

CREATE TABLE t (val halfvec(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[-1,-2,-3]'), (NULL);
CREATE INDEX ON t USING hnsw (val halfvec_l2_ops) WITH (quantization = 'binary');

SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> '[3,3,3]') t2;

DROP TABLE t;

-- more dimensions than hnsw allows without quantization, and changing the option takes effect at the next build

CREATE TABLE t (val vector(3000));
INSERT INTO t (val) SELECT array_fill(i, ARRAY[3000])::vector FROM generate_series(-2, 2) i;
CREATE INDEX idx ON t USING hnsw (val vector_l2_ops) WITH (quantization = 'binary');
WITH candidates AS MATERIALIZED (SELECT val, val <-> array_fill(1, ARRAY[3000])::vector AS distance FROM t ORDER BY distance LIMIT 10)
SELECT vector_dims(val), (val::real[])[1] FROM candidates ORDER BY distance + 0, (val::real[])[1] LIMIT 3;
ALTER INDEX idx SET (quantization = 'none');
INSERT INTO t (val) VALUES (array_fill(3, ARRAY[3000])::vector);
WITH candidates AS MATERIALIZED (SELECT val, val <-> array_fill(1, ARRAY[3000])::vector AS distance FROM t ORDER BY distance LIMIT 10)
SELECT (val::real[])[1] FROM candidates ORDER BY distance + 0, (val::real[])[1] LIMIT 3;
VACUUM t;
REINDEX INDEX idx;
DROP TABLE t;

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[1,2,3]'), ('[1,1,1]'), ('[-1,-2,-3]');
CREATE INDEX idx ON t USING hnsw (val vector_l2_ops);
ALTER INDEX idx SET (quantization = 'binary');
INSERT INTO t (val) VALUES ('[1,2,4]');
WITH candidates AS MATERIALIZED (SELECT val, val <-> '[3,3,3]' AS distance FROM t ORDER BY distance LIMIT 10)
SELECT val FROM candidates ORDER BY distance + 0;
REINDEX INDEX idx;
WITH candidates AS MATERIALIZED (SELECT val, val <-> '[3,3,3]' AS distance FROM t ORDER BY distance LIMIT 10)
SELECT val FROM candidates ORDER BY distance + 0;
DROP TABLE t;

-- relaxed_order fills LIMIT with a selective filter

CREATE TABLE t (i int, val vector(3));
SELECT setseed(0.5);
INSERT INTO t (i, val) SELECT i, ARRAY[random() - 0.5, random() - 0.5, random() - 0.5] FROM generate_series(1, 1000) i;
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantization = 'binary');
SET hnsw.ef_search = 10;
SET hnsw.iterative_scan = relaxed_order;
SELECT COUNT(*) FROM (SELECT i FROM t WHERE i % 50 = 0 ORDER BY val <-> '[0.1,0.1,0.1]' LIMIT 10) t2;
RESET hnsw.iterative_scan;
RESET hnsw.ef_search;
DROP TABLE t;

-- errors

CREATE TABLE t (val vector(3), s sparsevec(3), b bit(3));
CREATE INDEX ON t USING hnsw (s sparsevec_l2_ops) WITH (quantization = 'binary');
CREATE INDEX ON t USING hnsw (b bit_hamming_ops) WITH (quantization = 'binary');
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantization = 'pq');
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantization = 'binary');
SELECT * FROM t ORDER BY val <-> '[1,2]' LIMIT 1;
DROP TABLE t;
