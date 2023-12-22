select 
  engine
from 
  system.tables
where
  database = '{{ db_name }}'
and 
  name = '{{ table_name }}';
