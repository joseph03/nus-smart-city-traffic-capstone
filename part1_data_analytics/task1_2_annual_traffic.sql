-- Task 1.2: Annual traffic trends, 2012-2017

SELECT
    strftime('%Y', date_time) AS year,
    SUM(traffic_volume) AS total_traffic_volume
FROM traffic_clean
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Year-on-year change

WITH yearly_traffic AS (
    SELECT
        strftime('%Y', date_time) AS year,
        SUM(traffic_volume) AS total_traffic_volume
    FROM traffic_clean
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)

SELECT
    curr.year,
    curr.total_traffic_volume,
    prev.total_traffic_volume AS previous_year_volume,
    curr.total_traffic_volume - prev.total_traffic_volume AS change_from_previous_year,
    ROUND(
        100.0 * (curr.total_traffic_volume - prev.total_traffic_volume) 
        / prev.total_traffic_volume, 
        2
    ) AS percentage_change
FROM yearly_traffic curr
LEFT JOIN yearly_traffic prev 
    ON CAST(curr.year AS INTEGER) = CAST(prev.year AS INTEGER) + 1
ORDER BY curr.year;
