create temporary table coords (
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select stadt_x as x, stadt_y as y from coord_mapping_stadt_passive;
