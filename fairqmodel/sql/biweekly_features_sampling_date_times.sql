-- expand sampling dates on all 1-hour date_times
create temporary table sampling_date_times Engine=Memory as
-- sampling dates from passive samplers
with sampling_dates as (
select distinct
	date,
	date fdate
from
	fairq_raw.passive_samplers
where type = 'bi-weekly'
)
--- create date_time for joining and keep sampling date for later group by
select
toDateTime(date) date_time,
fdate as sampling_date
from
sampling_dates
order by date_time with fill STEP INTERVAL 1 HOUR INTERPOLATE (sampling_date as sampling_date);

