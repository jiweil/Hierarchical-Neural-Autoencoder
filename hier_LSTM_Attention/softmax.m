function[total_cost,grad]=softmax(result,docbatch,parameter)
    total_cost=0;
    grad.soft_W=zeroMatrix(size(parameter.soft_W));
    step_size=1;
    num_word=0;
    N=size(docbatch.target_sen_matrix,1);
    zeroState=zeroMatrix([parameter.hidden,N]);
    for ll=1:parameter.layer_num
        grad.source_h{ll,1}=zeroState;
        grad.source_c{ll,1}=zeroState;
        for sen_tt=1:length(result.Target_sen)-1
            grad.target_sen_h{ll,sen_tt}=zeroState;
            grad.target_sen_c{ll,sen_tt}=zeroState;
        end
    end
    for sen_tt=1:length(result.Target_sen)
        Word_List=docbatch.target_word{sen_tt}.Word;
        Word_Mask=docbatch.target_word{sen_tt}.Mask;
        num_word=num_word+length(find(Word_Mask==1));
        N=size(Word_List,1);
        T=size(Word_List,2);
        target_sen=result.Target_sen{sen_tt};
        N_examples=size(Word_List,1)*size(Word_List,2);
        predict_Words=reshape(Word_List,1,N_examples);
        mask=reshape(Word_Mask,1,N_examples);
        h_t=[];
        if sen_tt==1
            h_t=[h_t,result.source_sen{parameter.layer_num,1}];
        else
            dim=size(result.h_t_target_sen);
            h_t=[h_t,result.h_t_target_sen{dim(1),sen_tt-1}];
        end
        dim=size(target_sen.h_t_target_word);
        h_t=[h_t,[target_sen.h_t_target_word{parameter.layer_num,1:dim(2)-1}]];
        [cost,grad_softmax_h]=batchSoftmax(h_t,mask,predict_Words,parameter);
        total_cost=total_cost+cost;
        grad.soft_W=grad.soft_W+grad_softmax_h.soft_W;
        if sen_tt==1
            grad.source_h{parameter.layer_num,1}=grad_softmax_h.h(:,1:N);
        else 
            grad.target_sen_h{parameter.layer_num,sen_tt-1}=grad_softmax_h.h(:,1:N);
        end
        for i=1:T-1
            grad.ht{sen_tt}{1,i}=grad_softmax_h.h(:,N*i+1:N*(i+1));
        end
    end
    total_cost=total_cost/N;
    grad.soft_W=grad.soft_W/N;
    clear predict_Words; clear mask;
    clear grad_softmax_h;
end

function[cost,softmax_grad]=batchSoftmax(h_t,mask,predict_Words,parameter)
    unmaskedIds=find(mask==1);
    scores=parameter.soft_W*h_t;
    mx = max(scores,[],1);
    scores=bsxfun(@minus,scores,mx);
    scores=exp(scores);
    norms = sum(scores, 1);
    if length(find(mask==0))==0
        scores=bsxfun(@rdivide, scores, norms);
    else
        scores=bsxfun(@times,scores, mask./norms); 
    end
    scoreIndices = sub2ind(size(scores),predict_Words(unmaskedIds),unmaskedIds);
    cost=sum(-log(scores(scoreIndices)));
    scores(scoreIndices) =scores(scoreIndices) - 1;
    softmax_grad.soft_W=scores*h_t';  %(N_word*examples)*(examples*diemsnion)=N_word*diemsnion;
    softmax_grad.h=(scores'*parameter.soft_W)';%(diemsnion*N_word)*(N_word*examples)=dimension*examples
    clear scores;
    clear norms;
end
