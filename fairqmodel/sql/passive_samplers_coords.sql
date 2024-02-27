create temporary table coords (
    `station_id` Int16,
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select
    0 as station_id,
    stadt_x as x,
    stadt_y as y
from
    coord_mapping_stadt_passive;
