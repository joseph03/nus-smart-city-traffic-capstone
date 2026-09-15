-- Task 1.3a: check duplicate 2016 labour day and new year

SELECT *
FROM traffic_clean
WHERE date_time IN (
    '2016-09-05 00:00:00',
    '2016-01-01 00:00:00'
)
ORDER BY date_time;
