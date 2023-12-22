select
  model_id
from
  models_final
where `domain` = %(model_type)s
and pollutant = %(depvar)s;
