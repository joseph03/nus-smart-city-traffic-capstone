-- Task 1.3: Temperature around holidays

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
