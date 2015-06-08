classdef Batch < handle
    properties
        MaxLenSource=-1;
        MaxLenTarget=-1;
        MaxLen=-1;
        Word=[];
        Delete={};
        Left={};
        Mask={};
        Label=[];
        SourceLength=[];
        N_word=0;
    end
end
        
        
