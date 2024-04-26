create temporary table traffic Engine=Memory as
with traffic_avg_at_samplers as (
select
sampling_date,
x,
y,
avg(value) * 24 kfz_per_24h
from
fairq_features.traffic_model_predictions_passive_samplers
inner join
sampling_date_times using(date_time)
where model_id = 26 --q_kfz
group by x, y, sampling_date
)
select
sampling_date date,
x,
y,
toUInt32(kfz_per_24h * scaling) kfz_per_24h
from
traffic_avg_at_samplers
inner join
fairq_features.traffic_model_scaling tms using(x, y)
