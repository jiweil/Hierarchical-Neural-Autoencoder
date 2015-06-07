function[total_cost,grad]=softmax(topHidVecs,batch,parameter)
    MaxLenSource=batch.MaxLenSource;
    MaxLenTarget=batch.MaxLenTarget;
    MaxLen=batch.MaxLen;
    Word=batch.Word;
    Mask=batch.Mask;
    
    step_size=ceil(500/size(Word,1));%calculate softmax in bulk
    total_cost=0;
    grad.soft_W=zeroMatrix(size(parameter.soft_W));
    step_size=1;
    for Begin=MaxLenSource+1:step_size:MaxLen
        End=Begin+step_size-1;
        if End>MaxLen
            End=MaxLen;
        end
        N_examples=size(Word,1)*(End-Begin+1);
        predict_Words=reshape(Word(:,Begin:End),1,N_examples);
        mask=reshape(Mask(:,Begin:End),1,N_examples);%remove positions with no words
        h_t=[topHidVecs{:,Begin-1:End-1}];
        [cost,grad_softmax_h]=batchSoftmax(h_t,mask,predict_Words,parameter);
        total_cost=total_cost+cost;
        grad.soft_W=grad.soft_W+grad_softmax_h.soft_W;
        for j=Begin:End
            grad.ht{1,j-MaxLenSource}=grad_softmax_h.h(:,(j-Begin)*size(Word,1)+1:(j-Begin+1)*size(Word,1)); 
        end
    end
    total_cost=total_cost/size(Word,1);
    clear predict_Words; clear mask;
    clear grad_softmax_h;
end

function[cost,softmax_grad]=batchSoftmax(h_t,mask,predict_Words,parameter)
%softmax matrix
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
