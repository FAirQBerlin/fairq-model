create temporary table stadtstruktur Engine=Memory as
select
x,
y,
gewaesser,
gruenflaeche + wald gruenflaeche,
infrastruktur,
wohnnutzung,
density,
traffic_intensity_kfz traffic_intensity,
nox_h_15 + nox_i_15 nox_h,
nox_v_gn15 nox_v
from
statdstruktur_features
where (x, y) in (select stadt_x, stadt_y from coord_mapping_stadt_passive);
