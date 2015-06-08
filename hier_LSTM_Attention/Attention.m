function[result]=Attention(docbatch,result,h_t,parameter,sen_tt)
    % attention models, linking current target sentences to all input source sentences
    N=size(docbatch.source_sen_matrix,1);
    T=size(docbatch.source_sen_matrix,2);
    result.attention_input{sen_tt}=[result.source_each_sen;repmat(h_t,1,T)];
    % concatenate source sentence embeddings with target
    result.attention_vector{sen_tt}=tanh(parameter.Attention_W*result.attention_input{sen_tt});
    % compositions
    scores=parameter.Attention_U'*result.attention_vector{sen_tt};
    % linking scors 
    Matrix=reshape(scores,N,T);
    Matrix=exp(Matrix);
    Matrix=Matrix.*docbatch.source_sen_mask;
    norms=sum(Matrix,2);
    result.scores{sen_tt}=bsxfun(@rdivide,Matrix, norms);
    % normalization
    result.dA{sen_tt}=gpuArray();
    for i=1:N
        result.dA{sen_tt}=[result.dA{sen_tt};diag(result.scores{sen_tt}(i,:))-result.scores{sen_tt}(i,:)'*result.scores{sen_tt}(i,:)];
        % calculate derivatives for later backward propagation usage
    end
    scores=reshape(result.scores{sen_tt},1,N*T);

    M=bsxfun(@times,result.source_each_sen,scores);
    % multiply weights with source embeddings
    result.sum_vector{sen_tt}=zeroMatrix([parameter.hidden,N]);
    for i=1:T
        result.sum_vector{sen_tt}=result.sum_vector{sen_tt}+M(:,(i-1)*N+1:i*N);
        % weighted sum
    end
    clear Matrix;
    clear norms;
    clear scores;
    clear M;
end
