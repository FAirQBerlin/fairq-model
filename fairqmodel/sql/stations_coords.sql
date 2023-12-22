create temporary table coords (
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select
    stadt_x as x,
    stadt_y as y
from
    coord_mapping_stadt_station
inner join
    stations_for_predictions using station_id
where is_active in %(only_active_stations)s;
