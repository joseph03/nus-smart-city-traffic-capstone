
SELECT
    strftime('%Y', date_time) AS year,
    strftime('%m', date_time) AS month,
    COUNT(*) AS records
FROM traffic_clean
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year, month
ORDER BY year, month;


SELECT
    strftime('%Y', date_time) AS year,
    COUNT(DISTINCT date(date_time)) AS days_present
FROM traffic_clean
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;


