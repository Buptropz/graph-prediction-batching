WITH batches AS MATERIALIZED (
    SELECT CASE WHEN ? THEN 0 ELSE request_id END AS batch_id,
           list(pos ORDER BY pos) AS positions
    FROM requests
    GROUP BY 1
), evaluated AS MATERIALIZED (
    SELECT positions, graph_scores(positions) AS probabilities
    FROM batches
), expanded AS (
    SELECT unnest(positions) AS pos, unnest(probabilities) AS p
    FROM evaluated
)
SELECT r.pos, r.request_id, r.paper_id, e.p,
       CAST(list_position(e.p, list_max(e.p)) - 1 AS INTEGER) AS topic
FROM expanded e JOIN requests r USING (pos)
ORDER BY r.pos
