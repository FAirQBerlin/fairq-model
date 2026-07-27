SELECT model_object, model_object_residuals, pollutant, description, description_residuals FROM model_description
where model_id = %(id)s;
