function[]=try()
A=randSimpleMatrix([500,100000]);
B=randSimpleMatrix([500,1]);

tic
A'*B
toc
end
