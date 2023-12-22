create temporary table station_avg
ENGINE = Memory as
-- Get historical predictions (combining two different sources)
with station_preds_or_data as (
    -- Latest predictions for newer dates for which we don't have entries in
    -- model_predictions_stations_lags yet: get them from model_predictions_stations_latest
  select
    date_time,
    station_id,
    station_type,
  -- long to wide:
    anyIf(toNullable(value), model_id = (select model_id from fairq_output.models_final where pollutant = 'no2' and `domain` = 'temporal')) as no2_filled,
    anyIf(toNullable(value), model_id = (select model_id from fairq_output.models_final where pollutant = 'pm10' and `domain` = 'temporal')) as pm10_filled,
    anyIf(toNullable(value), model_id = (select model_id from fairq_output.models_final where pollutant = 'pm25' and `domain` = 'temporal')) AS pm25_filled
  from
    fairq_output.model_predictions_stations_latest final
  left join
    (select substring(id, 4, 3) station_id, stattyp station_type from fairq_raw.stadtstruktur_measuring_stations_processed
      where id in (select id from stations_for_predictions)) station_types using(station_id)
  where date_time > (select max(date_time) from fairq_output.model_predictions_stations_lags final)
  and date_time >= toDateTime(%(date_time_min)s, 'UTC')
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
  group by date_time, station_id, station_type
  union all
  -- unions with past predicted station data
  -- In case we already have a historical prediction in model_predictions_stations_lags, we use them
  select
    date_time,
    station_id,
    station_type,
    anyIf(toNullable(value), model_id = (select max(model_id) from fairq_output.model_predictions_stations_lags where model_id in
    (select model_id from fairq_output.model_description where pollutant = 'no2'))) as no2_filled,
    anyIf(toNullable(value), model_id = (select max(model_id) from fairq_output.model_predictions_stations_lags where model_id in
    (select model_id from fairq_output.model_description where pollutant = 'pm10'))) as pm10_filled,
    anyIf(toNullable(value), model_id = (select max(model_id) from fairq_output.model_predictions_stations_lags where model_id in
    (select model_id from fairq_output.model_description where pollutant = 'pm25'))) AS pm25_filled
  from
    fairq_output.model_predictions_stations_lags final
  left join
    (select substring(id, 4, 3) station_id, stattyp station_type from fairq_raw.stadtstruktur_measuring_stations_processed
      where id in (select id from stations_for_predictions)) station_types using(station_id)
  where date_time >= toDateTime(%(date_time_min)s, 'UTC')
  and date_time <= toDateTime(%(date_time_max)s, 'UTC')
  group by date_time, station_id, station_type
),
final_scaling as (
-- get the final scaling factor via cross join to the station data
  select
  date_time,
  kkfz_per_24h,
  scaling_stadtrand,
  stattyp_verkehr,
  -- compute the final scaling as diff between how much 'stadtrand' the station minus station type verkehr
  -- e.g. stations of type verkehr (1) get, for a cell with much traffic (scaling_stadtrand = 0), a 1: abs(0-1)
  -- e.g. stations of type verkehr (1) get, for a cell without traffic (scaling_stadtrand = 1), a 0: abs(1-1)
  -- e.g. stations of type stadtrand (0) get, for a cell with low traffic (scaling_stadtrand = 0.2), a 0.8: abs(0-0.2)
  abs(scaling_stadtrand - stattyp_verkehr) scaling_final,
  avg_pm10,
  avg_pm25,
  avg_no2
  from scaling_kfz_station
  cross join
  (  select
      date_time,
      station_type = 'Verkehr' as stattyp_verkehr,
      round(avg(pm10_filled), 1) as avg_pm10,
      round(avg(pm25_filled), 1) as avg_pm25,
      round(avg(no2_filled), 1) as avg_no2
    from
      station_preds_or_data
    group by stattyp_verkehr, date_time
    order by date_time desc, stattyp_verkehr) station
  ),
-- Get all selected coordinates and their kkfz values from Verkehrsmengenkarte"
coords_kkfz_mapping as (
  select
  toInt32(x) x,
  toInt32(y) y,
  toInt32(round(kfz_per_24h / 1000)) kkfz_per_24h
  from traffic_model_scaling
  where (x, y) in (select x, y from coords)
),
-- aggregate the avg of the background stations and the
-- avg of the traffic station to one weighted avg with the scaling factor
pre_result as (
  select
  date_time,
  kkfz_per_24h,
  sum(scaling_final * avg_pm10) wavg_pm10,
  sum(scaling_final * avg_pm25) wavg_pm25,
  sum(scaling_final * avg_no2) wavg_no2
  from
    final_scaling
  group by date_time, kkfz_per_24h
)
-- map to all selected x, y coords and compute the lags
select
  date_time,
  x,
  y,
  wavg_pm10,
  wavg_pm25,
  wavg_no2,
  lagInFrame(wavg_pm10, 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la1,
  lagInFrame(wavg_pm10, 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la2,
  lagInFrame(wavg_pm10, 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la3,
  lagInFrame(wavg_pm10, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la4,
  lagInFrame(wavg_pm10, 5) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la5,
  lagInFrame(wavg_pm10, 6) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la6,
  lagInFrame(wavg_pm10, 7) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la7,
  lagInFrame(wavg_pm10, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la8,
  lagInFrame(wavg_pm10, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la24,
  lagInFrame(wavg_pm10, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la48,
  lagInFrame(wavg_pm10, 72) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm10_la72,
  lagInFrame(wavg_pm25, 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la1,
  lagInFrame(wavg_pm25, 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la2,
  lagInFrame(wavg_pm25, 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la3,
  lagInFrame(wavg_pm25, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la4,
  lagInFrame(wavg_pm25, 5) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la5,
  lagInFrame(wavg_pm25, 6) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la6,
  lagInFrame(wavg_pm25, 7) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la7,
  lagInFrame(wavg_pm25, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la8,
  lagInFrame(wavg_pm25, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la24,
  lagInFrame(wavg_pm25, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la48,
  lagInFrame(wavg_pm25, 72) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_pm25_la72,
  lagInFrame(wavg_no2, 1) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la1,
  lagInFrame(wavg_no2, 2) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la2,
  lagInFrame(wavg_no2, 3) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la3,
  lagInFrame(wavg_no2, 4) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la4,
  lagInFrame(wavg_no2, 5) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la5,
  lagInFrame(wavg_no2, 6) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la6,
  lagInFrame(wavg_no2, 7) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la7,
  lagInFrame(wavg_no2, 8) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la8,
  lagInFrame(wavg_no2, 24) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la24,
  lagInFrame(wavg_no2, 48) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la48,
  lagInFrame(wavg_no2, 72) OVER (PARTITION BY (x, y) ORDER BY date_time ASC ROWS
             BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as wavg_no2_la72
from
  coords_kkfz_mapping
left join
  pre_result using(kkfz_per_24h);
