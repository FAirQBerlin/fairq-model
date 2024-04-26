create table biweekly_features_all
(
    `id` String,
    `date` Date,
    `x` UInt32,
    `y` UInt32,
    `no2` Nullable(UInt16),
    `no2_grid` Nullable(UInt16),
    `kfz_per_24h` UInt32,
    `wind_direction` Nullable(UInt8),
    `wind_speed` Nullable(Float64),
    `wind_speed_max` Nullable(Float64),
    `wind_speed_70q` Float64,
    `wind_speed_30q` Float64,
    `temperature` Nullable(Float64),
    `temperature_max` Nullable(Float64),
    `temperature_min` Nullable(Float64),
    `temperature_70q` Float64,
    `temperature_30q` Float64,
    `precipitation` Nullable(UInt16),
    `pressure_msl` Nullable(UInt16),
    `pressure_msl_max` Nullable(UInt16),
    `pressure_msl_min` Nullable(UInt16),
    `pressure_msl_70q` UInt16,
    `pressure_msl_30q` UInt16,
    `sunshine` Nullable(UInt16),
    `cloud_cover` Nullable(UInt8),
    `cloud_cover_70q` UInt8,
    `cloud_cover_30q` UInt8,
    `gewaesser` Float64,
    `gruenflaeche` Float64,
    `infrastruktur` Float64,
    `wohnnutzung` Float64,
    `density` Float64,
    `traffic_intensity` Float64,
    `nox_h` Float64,
    `nox_v` Float64
)
ENGINE = ReplacingMergeTree
ORDER BY (date, x, y) as
select
id,
traffic.date date,
traffic.x x,
traffic.y y,
no2,
no2_grid,
kfz_per_24h,
wind_direction,
wind_speed,
wind_speed_max,
wind_speed_70q,
wind_speed_30q,
temperature,
temperature_max,
temperature_min,
temperature_70q,
temperature_30q,
precipitation,
pressure_msl,
pressure_msl_max,
pressure_msl_min,
pressure_msl_70q,
pressure_msl_30q,
sunshine,
cloud_cover,
cloud_cover_70q,
cloud_cover_30q,
gewaesser,
gruenflaeche,
infrastruktur,
wohnnutzung,
density,
traffic_intensity,
nox_h,
nox_v
from
traffic
inner join
dwd on traffic.x = dwd.x and traffic.y = dwd.y and traffic.date = dwd.date
inner join
no2 on traffic.x = no2.x and traffic.y = no2.y and traffic.date = no2.date
left join
no2_grid on traffic.x = no2_grid.x and traffic.y = no2_grid.y and traffic.date = no2_grid.date
inner join
stadtstruktur on dwd.x = stadtstruktur.x and dwd.y = stadtstruktur.y
SETTINGS join_use_nulls = 1
;
