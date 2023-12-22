SELECT model_object, model_object_residuals, pollutant FROM model_description
where model_id = %(id)s;
