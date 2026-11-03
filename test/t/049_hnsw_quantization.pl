use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Initialize node
my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

my $dim = 128;
my $array = join(',', ('2 * random() - 1') x $dim);

$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", qq(
	SELECT setseed(0.42);
	CREATE TABLE tst (i serial, v vector($dim), h halfvec($dim));
	INSERT INTO tst (v) SELECT ARRAY[$array]::vector FROM generate_series(1, 3000);
	UPDATE tst SET h = v::halfvec;
	CREATE TABLE queries AS SELECT i, ARRAY[$array]::vector AS q FROM generate_series(1, 10) i;
));

for my $opclass ("vector_l2_ops", "vector_cosine_ops", "vector_ip_ops", "halfvec_l2_ops")
{
	my $col = $opclass =~ /^halfvec/ ? "h" : "v";
	my $op = $opclass =~ /cosine/ ? "<=>" : $opclass =~ /_ip_/ ? "<#>" : "<->";
	my $q = $col eq "h" ? "q.q::halfvec" : "q.q";
	my $first = $col eq "h" ? "(SELECT q::halfvec FROM queries WHERE i = 1)" : "(SELECT q FROM queries WHERE i = 1)";

	$node->safe_psql("postgres", qq(
		CREATE INDEX idx ON tst USING hnsw ($col $opclass) WITH (quantization = 'binary');
	));

	# Inserts after the build use the same quantization
	$node->safe_psql("postgres", qq(
		INSERT INTO tst (v) SELECT ARRAY[$array]::vector FROM generate_series(1, 500);
		UPDATE tst SET h = v::halfvec WHERE h IS NULL;
	));

	my $recall = $node->safe_psql("postgres", qq(
		SET enable_indexscan = off;
		CREATE TEMP TABLE exact AS SELECT q.i AS qi, (SELECT array_agg(i) FROM
			(SELECT i FROM tst ORDER BY $col $op $q LIMIT 10) t) AS ids FROM queries q;
		SET enable_indexscan = on;
		SET enable_seqscan = off;
		SET hnsw.ef_search = 400;
		SELECT count(*) / 100.0 FROM queries q JOIN exact e ON e.qi = q.i, LATERAL
			(WITH candidates AS MATERIALIZED (SELECT i, $col $op $q AS distance FROM tst ORDER BY distance LIMIT 400)
			 SELECT i FROM candidates ORDER BY distance + 0 LIMIT 10) a WHERE a.i = ANY (e.ids);
	));
	cmp_ok($recall, ">=", 0.9, "$opclass recall with a CTE re-sort of 400 candidates");

	my $count = $node->safe_psql("postgres", qq(
		SET enable_seqscan = off;
		SET hnsw.ef_search = 10;
		SET hnsw.iterative_scan = relaxed_order;
		SELECT count(*) FROM (SELECT i FROM tst WHERE i % 200 = 0
			ORDER BY $col $op $first LIMIT 10) t;
	));
	is($count, 10, "$opclass relaxed_order fills LIMIT with a selective filter");

	$node->safe_psql("postgres", "DROP INDEX idx; DELETE FROM tst WHERE i > 3000;");
}

done_testing();
