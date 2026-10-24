use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Initialize node
my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

my $dim = 64;
my $array = join(',', ('2 * random() - 1') x $dim);

$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", qq(
	SELECT setseed(0.42);
	CREATE TABLE tst (i serial, v bit($dim));
	INSERT INTO tst (v) SELECT binary_quantize(ARRAY[$array]::vector) FROM generate_series(1, 4000);
	CREATE TABLE queries AS SELECT i, ARRAY[$array]::vector AS q FROM generate_series(1, 10) i;
	CREATE INDEX idx ON tst USING hnsw (v bit_vector_ip_ops);
));

# Check the plan of the same query the recall test measures
my $plan = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	EXPLAIN SELECT a.d FROM queries q, LATERAL
		(SELECT v <#> q.q AS d FROM tst ORDER BY d LIMIT 20) a;
));
like($plan, qr/Index Scan using idx/);

# Count ties at the cutoff as matches
my $recall = $node->safe_psql("postgres", qq(
	SET enable_indexscan = off;
	CREATE TEMP TABLE exact AS SELECT q.i, (SELECT max(d) FROM
		(SELECT v <#> q.q AS d FROM tst ORDER BY d LIMIT 20) t) AS cutoff FROM queries q;
	SET enable_indexscan = on;
	SET enable_seqscan = off;
	SET hnsw.ef_search = 400;
	SELECT count(*) FILTER (WHERE a.d <= e.cutoff) / 200.0
	FROM queries q JOIN exact e USING (i), LATERAL
		(SELECT v <#> q.q AS d FROM tst ORDER BY d LIMIT 20) a;
));
cmp_ok($recall, ">=", 0.95);

for my $mode ("relaxed_order", "strict_order")
{
	my $count = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.ef_search = 10;
		SET hnsw.iterative_scan = $mode;
		SELECT count(*) FROM (SELECT i FROM tst WHERE i % 200 = 0
			ORDER BY v <#> (SELECT q FROM queries LIMIT 1) LIMIT 10) t;
	));
	is($count, 10, "$mode fills LIMIT with a selective filter");
}

done_testing();
