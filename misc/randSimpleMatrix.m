function [result]=randSimpleMatrix(size)
    result=rand(size,'double', 'gpuArray');
end
