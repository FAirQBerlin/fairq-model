create temporary table coords (
    `x` Int32,
    `y` Int32
) ENGINE = Memory as
select x, y from coords_{mode}_batches where batch = %(batch)s;
