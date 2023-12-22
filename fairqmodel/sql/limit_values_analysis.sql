select
station_id,
date_time,
date_time_forecast,
value as pred,
pm25_filled as pm25,
pm10_filled as pm10,
no2_filled as no2
from
model_predictions_temporal_tweak_values
left join
fairq_features.messstationen_filled using(station_id, date_time)
where model_id = %(model_id)s and station_id in ('117', '124', '143', '174')
order by station_id, date_time;
