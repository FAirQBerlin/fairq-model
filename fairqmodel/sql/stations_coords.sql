create temporary table coords (
    `station_id` Int16,
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select
    station_id,
    stadt_x as x,
    stadt_y as y
from
    coord_mapping_stadt_station
inner join
    stations_for_predictions using station_id
where is_active in %(only_active_stations)s;
