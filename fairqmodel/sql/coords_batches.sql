create temporary table coords (
    `station_id` Int16,
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
with station_ids as (
  select
    toUInt16(station_id) station_id,
    stadt_x x,
    stadt_y y
  from
    coord_mapping_stadt_station
  where coord_mapping_stadt_station.station_id in (
    select station_id from stations_for_predictions where is_active
    )
)
select
    station_id,
    x,
    y
from
    coords_{mode}_batches
left join
    station_ids using(x, y)
where batch = %(batch)s;
