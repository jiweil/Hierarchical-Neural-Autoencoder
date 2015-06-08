function[sum_vector]=Attention_Decode(docbatch,result,h_t,parameter,Index)
        
    N=length(Index);
    T=size(docbatch.source_sen_matrix,2);
    source_each_sen=[];
    for i=1:T
        index=N*(T-1)+Index;
        source_each_sen=[source_each_sen,result.source_each_sen(:,index)];
    end

    attention_input=[source_each_sen;repmat(h_t,1,T)];
    attention_vector=tanh(parameter.Attention_W*attention_input);
    scores=parameter.Attention_U'*attention_vector;
    
    Matrix=reshape(scores,N,T);
    Matrix=exp(Matrix);
    Matrix=Matrix.*docbatch.source_sen_mask(Index,:);
    norms=sum(Matrix,2);
    scores=bsxfun(@rdivide,Matrix, norms);
    scores=reshape(scores,1,N*T);

    M=bsxfun(@times,source_each_sen,scores);
    sum_vector=zeroMatrix([parameter.hidden,N]);
    for i=1:T
        sum_vector=sum_vector+M(:,(i-1)*N+1:i*N);
    end
    clear Matrix;
    clear norms;
    clear scores;
    clear M;
end
