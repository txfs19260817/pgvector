SELECT rabitq_quantize('[2]'::vector, 4) <-> '[3]'::vector;
SELECT '[3]'::vector <-> rabitq_quantize('[2]'::vector, 4);
SELECT rabitq_quantize('[0]'::vector, 4) <-> '[3]'::vector;
SELECT rabitq_quantize(NULL::vector) IS NULL;
SELECT b, round(rabitq_l2_distance(rabitq_quantize('[1,0,0]', b), rabitq_quantize('[1.05,0,0]', b))::numeric, 4) FROM unnest(ARRAY[1,2,4,8]) b;
SELECT rabitq_quantize('[1,2,3]'::vector, 3);
SELECT rabitq_quantize('[1,2,3]'::vector)::rabitq(2);
SELECT rabitq_quantize('[1,2,3]'::vector) <-> '[1,2]'::vector;
SELECT '\x020100013ff00000000000003ff000000000000000'::rabitq;
SELECT '\x010100013ff00000000000003ff0000000000000ff'::rabitq;
SELECT '\x010100013ff00000000000003ff0000000000000'::rabitq;
SELECT '\x010100017ff00000000000003ff000000000000000'::rabitq;
SELECT '\x010100013ff00000000000007e37e43c8800759c00'::rabitq;

-- encoding, I/O, and the self estimate at every bit width

DO $$
DECLARE d int; b int; v vector; w vector; c rabitq;
BEGIN
  FOREACH d IN ARRAY ARRAY[1,3,7,8,9,15,16,17,63,64,65,768,960,1536,2000,4000] LOOP
    SELECT ARRAY(SELECT sin(i)::real FROM generate_series(1,d) i)::vector INTO v;
    SELECT ARRAY(SELECT -sin(i)::real FROM generate_series(1,d) i)::vector INTO w;
    FOREACH b IN ARRAY ARRAY[1,2,4,8] LOOP
      c := rabitq_quantize(v,b);
      IF octet_length(rabitq_send(c)) <> 20 + (d*b+7)/8 THEN
        RAISE EXCEPTION 'unexpected padding';
      END IF;
      IF rabitq_send(c::text::rabitq) <> rabitq_send(c) THEN
        RAISE EXCEPTION 'text round trip mismatch';
      END IF;
      IF (c <-> v) > 0.001 THEN
        RAISE EXCEPTION 'incorrect self estimate';
      END IF;
      IF rabitq_l2_distance(c, c) >= rabitq_l2_distance(c, rabitq_quantize(w,b)) THEN
        RAISE EXCEPTION 'code distance prefers the opposite code';
      END IF;
    END LOOP;
  END LOOP;
END $$;

-- hnsw

SET enable_seqscan = off;

CREATE TABLE t (id int, v vector(3));
INSERT INTO t VALUES (1,'[1,2,3]'), (2,'[-1,-2,-3]'), (3,'[4,5,6]'), (4,'[0,0,0]'), (5,NULL);
CREATE INDEX idx ON t USING hnsw ((rabitq_quantize(v)::rabitq(3)) rabitq_l2_ops);

EXPLAIN (COSTS OFF) SELECT id FROM t ORDER BY '[1,2,3]'::vector <-> rabitq_quantize(v)::rabitq(3) LIMIT 1;
SELECT id FROM t ORDER BY rabitq_quantize(v)::rabitq(3) <-> '[1,2,3]'::vector LIMIT 1;
SELECT id FROM t ORDER BY rabitq_quantize(v)::rabitq(3) <-> '[1,2]'::vector LIMIT 1;
PREPARE scan (vector) AS SELECT id FROM t ORDER BY rabitq_quantize(v)::rabitq(3) <-> $1 LIMIT 1;
EXECUTE scan('[4,5,6]');
EXECUTE scan('[-1,-2,-3]');
DEALLOCATE scan;

SET hnsw.ef_search = 1;
SET hnsw.iterative_scan = relaxed_order;
SELECT COUNT(*) FROM (SELECT id FROM t ORDER BY rabitq_quantize(v)::rabitq(3) <-> '[1,2,3]'::vector LIMIT 50) s;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;

DELETE FROM t WHERE id = 1;
VACUUM t;
SELECT id FROM t ORDER BY rabitq_quantize(v)::rabitq(3) <-> '[4,5,6]'::vector LIMIT 1;
DROP TABLE t;

-- the largest code fits on an index page

CREATE TABLE t (v vector(4000));
INSERT INTO t SELECT ARRAY(SELECT sin(i)::real FROM generate_series(1, 4000) i)::vector;
CREATE INDEX ON t USING hnsw ((rabitq_quantize(v, 8)::rabitq(4000)) rabitq_l2_ops);
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY rabitq_quantize(v, 8)::rabitq(4000) <-> (SELECT v FROM t)) s;
DROP TABLE t;

RESET enable_seqscan;

-- query type without query functions

CREATE OPERATOR CLASS rabitq_invalid_ops FOR TYPE rabitq USING hnsw AS
	OPERATOR 1 <-> (rabitq, vector) FOR ORDER BY float_ops,
	FUNCTION 1 rabitq_l2_distance(rabitq, rabitq);
SELECT amvalidate(oid) FROM pg_opclass WHERE opcname = 'rabitq_invalid_ops';
DROP OPERATOR FAMILY rabitq_invalid_ops USING hnsw;
