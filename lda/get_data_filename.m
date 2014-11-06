%------------------------------------------------------------------------------
function file_name = get_data_filename(dir, class_id, do_normalize)
%------------------------------------------------------------------------------
file_name = [dir 'DATA-' int2str(class_id)];
if do_normalize
  file_name = [file_name '-norm'];
end
file_name = [file_name '.mat'];

