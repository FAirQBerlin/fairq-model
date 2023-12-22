select
 station_id id,
 stattyp
from
  fairq_raw.stadtstruktur_measuring_stations sms
inner join
  fairq_features.stations_for_predictions sfp on sms.id = sfp.id;

