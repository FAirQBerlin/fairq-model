select model_id, date_time_training_execution, pollutant, model_name, description, description_residuals
from model_description
where model_name like %(model_type)s and pollutant = %(depvar)s
order by model_id, pollutant, date_time_training_execution desc;

