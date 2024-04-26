create temporary table no2 Engine=Memory as
with station_avg as (
select concat('MC ',station_id) id, sampling_date date, toUInt16(avg(no2)) no2 from messstationen_filled
left join sampling_date_times using(date_time)
where isNotNull(sampling_date)
group by station_id, sampling_date
settings join_use_nulls=1
)
select
id, date, stadt_x x, stadt_y y, no2
from
station_avg
inner join
coord_mapping_stadt_passive cmsp using(id)
union all
select toString(cmsp.id) id, date, stadt_x x, stadt_y y, no2 from fairq_raw.passive_samplers ps
inner join
coord_mapping_stadt_passive cmsp on concat('MP ', toString(ps.id)) = cmsp.id
where type = 'bi-weekly';
