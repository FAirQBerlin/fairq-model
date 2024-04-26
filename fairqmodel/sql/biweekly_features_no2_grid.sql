create temporary table no2_grid Engine=Memory as
select
 sampling_date date,
 x,
 y,
 round(avg(value), 0) as no2_grid
from
 fairq_prod_output.model_predictions_grid_history
inner join
 sampling_date_times using(date_time)
where (x, y) in (select stadt_x, stadt_y from coord_mapping_stadt_passive)
and model_id in (select model_id from fairq_prod_output.model_description where pollutant = 'no2')
and date_time >= (select min(date_time) from sampling_date_times)
group by x, y, date;
