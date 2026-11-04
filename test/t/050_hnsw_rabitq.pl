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
my $expr = "rabitq_quantize(v, 4)::rabitq($dim)";

$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", qq(
	SELECT setseed(0.42);
	CREATE TABLE tst (i serial, v vector($dim));
	INSERT INTO tst (v) SELECT ARRAY[$array]::vector FROM generate_series(1, 4000);
	CREATE TABLE queries AS SELECT i, ARRAY[$array]::vector AS q FROM generate_series(1, 10) i;
));

# Recall of the estimated distance, counting ties at the cutoff as matches
sub test_recall
{
	my ($name) = @_;

	my $plan = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		EXPLAIN SELECT a.d FROM queries q, LATERAL
			(SELECT $expr <-> q.q AS d FROM tst ORDER BY d LIMIT 20) a;
	));
	like($plan, qr/Index Scan using idx/, "$name uses the index");

	my $recall = $node->safe_psql("postgres", qq(
		SET enable_indexscan = off;
		CREATE TEMP TABLE exact AS SELECT q.i, (SELECT max(d) FROM
			(SELECT $expr <-> q.q AS d FROM tst ORDER BY d LIMIT 20) t) AS cutoff FROM queries q;
		SET enable_indexscan = on;
		SET enable_seqscan = off;
		SET hnsw.ef_search = 400;
		SELECT count(*) FILTER (WHERE a.d <= e.cutoff + 1e-6) / 200.0 || ' ' || count(*)
		FROM queries q JOIN exact e USING (i), LATERAL
			(SELECT $expr <-> q.q AS d FROM tst ORDER BY d LIMIT 20) a;
	));
	my ($fraction, $rows) = split(" ", $recall);
	cmp_ok($fraction, ">=", 0.95, "$name recall");
	is($rows, 200, "$name rescans return every row");
}

for my $mode ("serial", "parallel", "spill")
{
	my $workers = $mode eq "serial" ? 0 : 2;
	my $memory = $mode eq "spill" ? "1MB" : "64MB";
	my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
		SET client_min_messages = DEBUG;
		SET max_parallel_maintenance_workers = $workers;
		SET min_parallel_table_scan_size = 0;
		SET maintenance_work_mem = '$memory';
		ALTER TABLE tst SET (parallel_workers = $workers);
		CREATE INDEX idx ON tst USING hnsw (($expr) rabitq_l2_ops);
	));
	is($ret, 0, "$mode build");
	like($stderr, qr/using [1-9]\d* parallel workers/) if $workers;
	like($stderr, qr/hnsw graph no longer fits/) if $mode eq "spill";
	test_recall($mode);
	$node->safe_psql("postgres", "DROP INDEX idx;");
}

$node->safe_psql("postgres", "CREATE INDEX idx ON tst USING hnsw (($expr) rabitq_l2_ops);");

my $plan = $node->safe_psql("postgres", qq(
	SET enable_seqscan = off;
	EXPLAIN SELECT i FROM tst WHERE i % 50 = 0 ORDER BY $expr <-> (SELECT q FROM queries LIMIT 1) LIMIT 10;
));
like($plan, qr/Index Scan using idx/, "filtered query uses the index");

for my $mode ("relaxed_order", "strict_order")
{
	my $count = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.ef_search = 10;
		SET hnsw.iterative_scan = $mode;
		SELECT count(*) FROM (SELECT i FROM tst WHERE i % 50 = 0
			ORDER BY $expr <-> (SELECT q FROM queries LIMIT 1) LIMIT 10) t;
	));
	is($count, 10, "$mode fills LIMIT with a selective filter");
}

# Candidates must contain the true nearest neighbors, not just the estimated ones.
# The estimator is centered at the origin, so offset data uses a fixed center.
my $offset = "array_fill(1000::real, ARRAY[$dim])::vector";
$node->safe_psql("postgres", qq(
	CREATE TABLE far AS SELECT i, v + $offset AS v FROM tst;
	CREATE TABLE far_queries AS SELECT i, q + $offset AS q FROM queries;
	CREATE INDEX far_idx ON far USING hnsw ((rabitq_quantize(v - $offset, 4)::rabitq($dim)) rabitq_l2_ops);
));
for my $case (["centered", "tst", "queries", "v", "q.q"], ["offset", "far", "far_queries", "v - $offset", "q.q - $offset"])
{
	my ($name, $table, $queries, $value, $query) = @$case;
	my $recall = $node->safe_psql("postgres", qq(
		CREATE TEMP TABLE truth AS SELECT q.i AS qi, t.i FROM $queries q, LATERAL
			(SELECT i FROM $table ORDER BY v <-> q.q LIMIT 10) t;
		SET enable_seqscan = off;
		SET hnsw.ef_search = 400;
		SELECT count(*) / 100.0 FROM $queries q, LATERAL (SELECT i FROM $table
			ORDER BY rabitq_quantize($value, 4)::rabitq($dim) <-> ($query) LIMIT 100) c, truth t
		WHERE t.qi = q.i AND t.i = c.i;
	));
	cmp_ok($recall, ">=", 0.9, "$name candidates contain the true neighbors");
}

# Inserts and vacuum use the code-to-code distance
$node->safe_psql("postgres", "TRUNCATE tst RESTART IDENTITY;");
$node->pgbench(
	"--no-vacuum --client=4 --transactions=1000",
	0,
	[qr{actually processed}],
	[qr{^$}],
	"concurrent inserts",
	{
		"050_hnsw_rabitq" => "INSERT INTO tst (v) VALUES (ARRAY[$array]::vector);"
	}
);
test_recall("insert");

$node->safe_psql("postgres", "DELETE FROM tst WHERE i % 4 <> 0;");
$node->safe_psql("postgres", "VACUUM tst;");
test_recall("vacuum");

done_testing();
