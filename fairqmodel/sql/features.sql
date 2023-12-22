With latest_traffic_preds as (
  select
 	  date_time,
	  x,
	  y,
	  value
  from
  -- either traffic_model_predictions_grid  --> predictions on grid
  -- or traffic_model_predictions_stations --> training / predictions at stations
	  traffic_model_predictions_{traffic_table_suffix} final
  where
	  model_id = (
      select
        model_id
      from
        traffic_models_final
      where depvar = 'q_kfz'
	)
	and (x, y) in (select toUInt32(x), toUInt32(y) from coords)
  and date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due due 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')

),

traffic_scaled as (
  select
	  date_time,
	  x,
	  y,
	  toNullable(scaling * value) as kfz_per_hour,
    lagInFrame(toNullable(kfz_per_hour), 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as kfz_per_hour_la4,
    lagInFrame(toNullable(kfz_per_hour), 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as kfz_per_hour_la8,
    lagInFrame(toNullable(kfz_per_hour), 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as kfz_per_hour_la24,
    lagInFrame(toNullable(kfz_per_hour), 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as kfz_per_hour_la48
  from
	  latest_traffic_preds lt
  inner join
    traffic_model_scaling tms
    on lt.x = tms.x
	and lt.y = tms.y
),

latest_velocity_preds AS (
  SELECT
    date_time,
    x,
    y,
    toNullable(value) AS velocity
  FROM
    traffic_model_predictions_{traffic_table_suffix} final
  WHERE model_id = (
    select model_id from traffic_models_final where depvar = 'v_kfz'
  )
  and date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due to 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
  and (x, y) in (select toUInt32(x), toUInt32(y) from coords)
),

dwd_obs_and_forecasts as (
  select
    date_time AS date_time,
    x,
    y,
    wind_direction,
    wind_speed,
    precipitation,
    temperature,
    cloud_cover,
    pressure_msl,
    sunshine
  from
    fairq_raw.dwd_forecasts_processed dwd final
  where date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due due 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
  union all
  select
    date_time,
    x,
    y,
    wind_direction_filled AS wind_direction,
    wind_speed_filled AS wind_speed,
    precipitation_filled AS precipitation,
    temperature_filled AS temperature,
    cloud_cover_filled AS cloud_cover,
    pressure_msl_filled AS pressure_msl,
    sunshine_filled AS sunshine
  from
    dwd_observations_filled
  where date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due due 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
  and date_time < (select min(date_time) as date_first_forecast from fairq_raw.dwd_forecasts_processed)
),

-- Latest dwd forecasts, mapped onto grid coordinates and station coordinates
coord_mapping_stadt_dwd as (
  select
    *
  from
    coord_mapping_stadt_dwd
  where (stadt_x, stadt_y) in (select x, y from coords)
),

dwd_data_mapped as (
  SELECT
    date_time,
    case when station_id = '' then NULL else station_id end as station_id,
    cmsd.stadt_x as x,
    cmsd.stadt_y as y,
    wind_direction,
		wind_speed,
		precipitation,
		temperature,
		cloud_cover,
		pressure_msl,
		sunshine,
    lagInFrame(wind_direction, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wind_direction_la4,
    lagInFrame(wind_direction, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wind_direction_la8,
    lagInFrame(wind_direction, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wind_direction_la24,
    lagInFrame(wind_direction, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wind_direction_la48,
    lagInFrame(precipitation, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as precipitation_la4,
    lagInFrame(precipitation, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as precipitation_la8,
    lagInFrame(precipitation, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as precipitation_la24,
    lagInFrame(precipitation, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as precipitation_la48,
    lagInFrame(pressure_msl, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as pressure_msl_la4,
    lagInFrame(pressure_msl, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as pressure_msl_la8,
    lagInFrame(pressure_msl, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as pressure_msl_la24,
    lagInFrame(pressure_msl, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as pressure_msl_la48
  FROM
    dwd_obs_and_forecasts dofc
  INNER JOIN
    coord_mapping_stadt_dwd cmsd
    on (dofc.x = cmsd.dwd_x) AND (dofc.y = cmsd.dwd_y)
  LEFT JOIN
    coord_mapping_stadt_station cmss
    on (cmsd.stadt_x = cmss.stadt_x) and (cmsd.stadt_y = cmss.stadt_y)
  where date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due due 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')

),

-- Latest CAMS forecasts, mapped onto grid coordinates
latest_cams_fast as (
  SELECT
    date_time,
    x,
    y,
    no2_filled no2,
    pm25_filled pm25,
    pm10_filled pm10
  FROM
    cams_all_latest_filled final
  INNER JOIN
    mapping_reprojection USING (lat_int, lon_int)
  where date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due to 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
),

cams_data_mapped as (
	with coord_mapping_stadt_cams as (
			select
        *
      from
        coord_mapping_stadt_cams
			where (stadt_x, stadt_y) in (select x, y from coords)
	)
	SELECT
    date_time,
    stadt_x as x,
    stadt_y as y,
    no2 AS cams_no2,
    pm25 AS cams_pm25,
    pm10 AS cams_pm10,
    lagInFrame(toNullable(cams_no2), 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la1,
    lagInFrame(toNullable(cams_no2), 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la2,
    lagInFrame(toNullable(cams_no2), 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la3,
    lagInFrame(toNullable(cams_no2), 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la4,
    lagInFrame(toNullable(cams_no2), 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la8,
    lagInFrame(toNullable(cams_no2), 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la24,
    lagInFrame(toNullable(cams_no2), 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_no2_la48,
    lagInFrame(toNullable(cams_pm10), 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la1,
    lagInFrame(toNullable(cams_pm10), 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la2,
    lagInFrame(toNullable(cams_pm10), 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la3,
    lagInFrame(toNullable(cams_pm10), 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la4,
    lagInFrame(toNullable(cams_pm10), 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la8,
    lagInFrame(toNullable(cams_pm10), 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la24,
    lagInFrame(toNullable(cams_pm10), 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm10_la48,
    lagInFrame(toNullable(cams_pm25), 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la1,
    lagInFrame(toNullable(cams_pm25), 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la2,
    lagInFrame(toNullable(cams_pm25), 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la3,
    lagInFrame(toNullable(cams_pm25), 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la4,
    lagInFrame(toNullable(cams_pm25), 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la8,
    lagInFrame(toNullable(cams_pm25), 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la24,
    lagInFrame(toNullable(cams_pm25), 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
                 BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as cams_pm25_la48
  FROM
    latest_cams_fast AS lc
  INNER JOIN
    coord_mapping_stadt_cams
    ON (lc.x = cams_x) AND (lc.y = cams_y)
  where date_time >= toDateTime(%(date_time_min)s, 'UTC') - interval 2 day -- due due 48 hours lag
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
),

-- measuring stations
measuring_stations as (
  select
    date_time,
    station_id,
    stadt_x as x,
    stadt_y as y,
    pm10_filled as pm10,
    pm25_filled as pm25,
    no2_filled as no2
    -- isNull(pm25) as is_pm25_filled,
    -- isNull(pm10) as is_pm10_filled,
    -- isNull(no2) as is_no2_filled
  from
    messstationen_filled ms final
  inner join
    coord_mapping_stadt_station
  on ms.x = station_x and ms.y = station_y
  where date_time >= toDateTime(%(date_time_min)s, 'UTC')
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
)

SELECT
  dwd.station_id as station_id,
  dwd.date_time as date_time,
  dwd.x as x,
  dwd.y as y,
  pm10,
  pm25,
  no2,
  dt.doy_scaled AS doy_scaled,
  toDayOfWeek(dt.date_time) AS weekday,
  toYear(dt.date_time) AS year,
  toHour(dt.date_time) AS hour,
  kfz_per_hour,
  velocity,
  cams_no2,
  cams_pm25,
  cams_pm10,
  wind_direction,
  wind_speed,
  precipitation,
  temperature,
  cloud_cover,
  pressure_msl,
  sunshine,
  type_school_holiday,
  is_public_holiday,
  grauflaeche,
  gewaesser,
  gruenflaeche,
  infrastruktur,
  mischnutzung,
  wald,
  wohnnutzung,
  density as building_density,
  traffic_intensity_kfz,
  nox_h_15 as nox_h,
  nox_i_15 as nox_i,
  nox_v_gn15 as nox_v,
  pm10_i_15 as pmx_i,
  kfz_per_hour_la4,
  kfz_per_hour_la8,
  kfz_per_hour_la24,
  kfz_per_hour_la48,
  cams_no2_la1,
  cams_no2_la2,
  cams_no2_la3,
  cams_no2_la4,
  cams_no2_la8,
  cams_no2_la24,
  cams_no2_la48,
  cams_pm10_la1,
  cams_pm10_la2,
  cams_pm10_la3,
  cams_pm10_la4,
  cams_pm10_la8,
  cams_pm10_la24,
  cams_pm10_la48,
  cams_pm25_la1,
  cams_pm25_la2,
  cams_pm25_la3,
  cams_pm25_la4,
  cams_pm25_la8,
  cams_pm25_la24,
  cams_pm25_la48,
  wavg_no2,
  (wavg_no2_la1 + wavg_no2_la2 + wavg_no2_la3 +
   wavg_no2_la4 + wavg_no2_la5 + wavg_no2_la6 +
   wavg_no2_la7 + wavg_no2_la8) / 8 as wavg_no2_la1to8,
  wavg_no2_la1,
  wavg_no2_la2,
  wavg_no2_la3,
  wavg_no2_la4,
  wavg_no2_la5,
  wavg_no2_la6,
  wavg_no2_la7,
  wavg_no2_la8,
  wavg_no2_la24,
  wavg_no2_la48,
  wavg_no2_la72,
  wavg_pm10,
  (wavg_pm10_la1 + wavg_pm10_la2 + wavg_pm10_la3 +
   wavg_pm10_la4 + wavg_pm10_la5 + wavg_pm10_la6 +
   wavg_pm10_la7 + wavg_pm10_la8) / 8 as wavg_pm10_la1to8,
  wavg_pm10_la1,
  wavg_pm10_la2,
  wavg_pm10_la3,
  wavg_pm10_la4,
  wavg_pm10_la5,
  wavg_pm10_la6,
  wavg_pm10_la7,
  wavg_pm10_la8,
  wavg_pm10_la24,
  wavg_pm10_la48,
  wavg_pm10_la72,
  wavg_pm25,
  (wavg_pm25_la1 + wavg_pm25_la2 + wavg_pm25_la3 +
   wavg_pm25_la4 + wavg_pm25_la5 + wavg_pm25_la6 +
   wavg_pm25_la7 + wavg_pm25_la8) / 8 as wavg_pm25_la1to8,
  wavg_pm25_la1,
  wavg_pm25_la2,
  wavg_pm25_la3,
  wavg_pm25_la4,
  wavg_pm25_la5,
  wavg_pm25_la6,
  wavg_pm25_la7,
  wavg_pm25_la8,
  wavg_pm25_la24,
  wavg_pm25_la48,
  wavg_pm25_la72,
  wind_direction_la4,
  wind_direction_la8,
  wind_direction_la24,
  wind_direction_la48,
  precipitation_la4,
  precipitation_la8,
  precipitation_la24,
  precipitation_la48,
  pressure_msl_la4,
  pressure_msl_la8,
  pressure_msl_la24,
  pressure_msl_la48
from
  dwd_data_mapped as dwd
inner join
  cams_data_mapped as cams
  on (dwd.date_time = cams.date_time) and (dwd.x = cams.x) and (dwd.y = cams.y)
inner join
  traffic_scaled AS ts
  ON (dwd.date_time = ts.date_time) AND (dwd.x = ts.x) AND (dwd.y = ts.y)
left join
  features_date_time AS dt
  ON dwd.date_time = dt.date_time
left join
  latest_velocity_preds AS lvp
  ON (dwd.date_time = lvp.date_time) AND (dwd.x = lvp.x) AND (dwd.y = lvp.y)
left join
  statdstruktur_features as land
  ON (dwd.x = land.x) AND (dwd.y = land.y)
left join
  measuring_stations ms
  on (dwd.date_time = ms.date_time) and (dwd.x = ms.x) and (dwd.y = ms.y)
left join
  station_avg savg on (dwd.date_time = savg.date_time) and (dwd.x = savg.x) and (dwd.y = savg.y)
where date_time >= toDateTime(%(date_time_min)s, 'UTC')
and date_time <= toDateTime(%(date_time_max)s, 'UTC')
;
