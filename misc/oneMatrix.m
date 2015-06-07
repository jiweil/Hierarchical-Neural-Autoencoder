function [result] = oneMatrix(size)
    result =ones(size, 'double', 'gpuArray');
end
