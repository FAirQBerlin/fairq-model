create temporary table coords (
    `station_id` Int16,
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select
    0 as station_id,
    x,
    y
from
    coords_{mode}_batches
where batch = %(batch)s;
