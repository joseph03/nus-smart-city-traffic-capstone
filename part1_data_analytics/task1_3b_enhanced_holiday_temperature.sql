-- Task 1.3: Original Temperature around holidays

SELECT
    strftime('%Y', date_time) AS year,
    holiday,
    date_time,
    temp,
    ROUND(temp - 273.15, 2) AS temp_celsius,
    traffic_volume
FROM traffic_clean
WHERE strftime('%Y', date_time) IN ('2015', '2016', '2017')
  AND holiday IN ('New Years Day', 'Labor Day')
ORDER BY year, holiday, date_time;

-- Task 1.3b: Enhanced Temperature around holidays

SELECT
    strftime('%Y', date_time) AS year,
    holiday,
    date_time,
    ROUND(AVG(temp), 2) AS temp,
    ROUND(AVG(temp) - 273.15, 2) AS temp_celsius,
    MAX(traffic_volume) AS traffic_volume
FROM traffic_clean
WHERE year IN ('2015', '2016', '2017')
  AND holiday IN ('New Years Day', 'Labor Day')
GROUP BY year, holiday, date_time
ORDER BY year, holiday, date_time;