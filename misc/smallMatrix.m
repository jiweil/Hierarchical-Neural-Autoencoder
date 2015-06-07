function [result] = smallMatrix(size)
    result =10^(-50)*ones(size, 'double', 'gpuArray');
end
