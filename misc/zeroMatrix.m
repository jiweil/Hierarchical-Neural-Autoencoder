function [result] = zeroMatrix(size)
    result = zeros(size, 'double', 'gpuArray');
end
