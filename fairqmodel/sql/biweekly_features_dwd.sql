create temporary table dwd Engine=Memory as
with dwd_obs_and_forecasts as (
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
  where date_time >= '2020-01-01'--(select min(toDateTime(sampling_date)) from sampling_date_times)
  and date_time < (select min(date_time) as date_first_forecast from fairq_raw.dwd_forecasts_processed)
),
dwd_aggregated as (
select
sampling_date,
x dwd_x,
y dwd_y,
avg(wind_speed * sin(wind_direction * pi() / 180)) wind_u,
avg(wind_speed * cos(wind_direction * pi() / 180)) wind_v,
avg(wind_speed) wind_speed_avg,
max(wind_speed) wind_speed_max,
quantiles(0.7)(wind_speed)[1] wind_speed_70q,
quantiles(0.3)(wind_speed)[1] wind_speed_30q,
avg(temperature) temperature_avg,
max(temperature) temperature_max,
min(temperature) temperature_min,
quantiles(0.7)(temperature)[1] temperature_70q,
quantiles(0.3)(temperature)[1] temperature_30q,
sum(precipitation) precipitation,
avg(pressure_msl) pressure_msl_avg,
max(pressure_msl) pressure_msl_max,
min(pressure_msl) pressure_msl_min,
quantiles(0.7)(pressure_msl)[1] pressure_msl_70q,
quantiles(0.3)(pressure_msl)[1] pressure_msl_30q,
sum(sunshine) / 60 sunshine,
avg(cloud_cover) cloud_cover_avg,
quantiles(0.7)(cloud_cover)[1] cloud_cover_70q,
quantiles(0.3)(cloud_cover)[1] cloud_cover_30q
from
dwd_obs_and_forecasts
inner join
sampling_date_times using(date_time)
group by sampling_date, x, y
)
select
sampling_date date,
stadt_x x,
stadt_y y,
toUInt8(modulo(360 + atan2(wind_u, wind_v) * 180 / pi(), 360)) wind_direction,
round(wind_speed_avg, 1) wind_speed,
round(wind_speed_max, 1) wind_speed_max,
round(wind_speed_70q, 1) wind_speed_70q,
round(wind_speed_30q, 1) wind_speed_30q,
round(temperature_avg, 1) temperature,
round(temperature_max, 1) temperature_max,
round(temperature_min, 1) temperature_min,
round(temperature_70q, 1) temperature_70q,
round(temperature_30q, 1) temperature_30q,
toUInt16(precipitation) precipitation,
toUInt16(pressure_msl_avg) pressure_msl,
toUInt16(pressure_msl_max) pressure_msl_max,
toUInt16(pressure_msl_min) pressure_msl_min,
toUInt16(pressure_msl_70q) pressure_msl_70q,
toUInt16(pressure_msl_30q) pressure_msl_30q,
toUInt16(sunshine) sunshine,
toUInt8(cloud_cover_avg) cloud_cover,
toUInt8(cloud_cover_70q) cloud_cover_70q,
toUInt8(cloud_cover_30q) cloud_cover_30q
from
dwd_aggregated
inner join
(select * from coord_mapping_stadt_dwd where (stadt_x, stadt_y) in (select stadt_x, stadt_y from coord_mapping_stadt_passive)) coords using(dwd_x, dwd_y)
;
