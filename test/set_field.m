function model = set_field(model, field, val)
if ~isfield(model, field)
  eval(['model.' field ' = val;']);
end
