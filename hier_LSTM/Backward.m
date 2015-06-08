function[grad]=Backward(current_batch,result,grad,parameter)
    grad=Backward_Target(current_batch,result,grad,parameter);
    % backward propagatin for target sentences
    grad=Backward_Source_Sen(current_batch,result,grad,parameter);
    % backward propagatin for sources sentences
end

function[grad]=Backward_Source_Sen(docbatch,result,grad,parameter)
    num_sentence=size(docbatch.target_sen_matrix,1);
    for ll=1:parameter.layer_num
        dh{ll}=grad.source_h{ll,1}; % dh
        dc{ll}=grad.source_c{ll,1}; % dc
    end
    N=size(docbatch.source_sen_matrix,1); 
    % number of documents in current batches
    T=size(docbatch.source_sen_matrix,2);
    % total time steps at sentence level
    zeroState=zeroMatrix([parameter.hidden,N]);
    clear grad.source_h;
    clear grad.source_c;
    d_source_sen=zeroMatrix([parameter.hidden,docbatch.num_of_source_sen]);
    for sen_tt=T:-1:1
        for ll=parameter.layer_num:-1:1
            if sen_tt==1
                c_t_1 =zeroState;
            else
                c_t_1=result.c_t_source_sen{ll,sen_tt-1};
            end
            c_t=result.c_t_source_sen{ll,sen_tt};
            lstm=result.lstms_source_sen{ll,sen_tt};
            dh{ll}(:,docbatch.source_delete{sen_tt})=0;
            dc{ll}(:,docbatch.source_delete{sen_tt})=0;
            % set values to zero for positions without words
            if sen_tt==1
                is_begin=1;
            else
                is_begin=0;
            end
            [lstm_grad]=lstmUnitGrad(lstm,c_t, c_t_1,dc{ll},dh{ll},ll,sen_tt,zeroState,parameter,parameter.Sen_S{ll},is_begin);
            % backward calculation for lstm unit at source sentence level
            lstm_grad.input(:,docbatch.source_delete{sen_tt})=0;
            dc{ll}(:,docbatch.source_delete{sen_tt})=0;
            dc{ll}=lstm_grad.dc;
            grad.Sen_S{ll}=grad.Sen_S{ll}+lstm_grad.W;

            lstm_grad.input(:,docbatch.source_delete{sen_tt})=0;
            dh{ll}=lstm_grad.input(end-parameter.hidden+1:end,:);
            dc{ll}=lstm_grad.dc;
            if ll~=1
                dh{ll-1}=dh{ll-1}+lstm_grad.input(1:parameter.hidden,:);
            else
                sen_index=docbatch.source_sen_matrix(:,sen_tt);
                d_source_sen(:,sen_index(docbatch.source_left{sen_tt}))=lstm_grad.input(1:parameter.hidden,docbatch.source_left{sen_tt});
                % if ll==1, gradient for emebddings at sentence level
            end
        end
    end
    clear dh;
    clear dc;
    clear lstm_grad;
    clear result.c_t_source_sen;
    clear result.lstms_source_sen;

    grad=DealSentenceSource(d_source_sen,docbatch,result,grad,parameter);
    % backward propagation to source sentences at word le
    for ll=1:parameter.layer_num
        grad.Word_S{ll}=grad.Word_S{ll}/num_sentence;
        grad.Sen_S{ll}=grad.Sen_S{ll}/num_sentence;
    end
    grad.vect=grad.vect/num_sentence;
end

function[grad]=DealSentenceSource(d_source_sen,docbatch,result,grad,parameter)
    % backward propagation to source sentences at word level
    wordCount = 0;
    for i=1:length(docbatch.source_smallBatch)
        source_smallBatch=docbatch.source_smallBatch{i};
        Word=source_smallBatch.Word;
        N=size(Word,1);
        % number of sentences for current batch
        T=size(Word,2);
        % total number of time steps
        zeroState=zeroMatrix([parameter.hidden,N]);
        for ll=1:parameter.layer_num
            dh{ll}=zeroState;
            dc{ll}=zeroState;
        end
        for word_tt=T:-1:1
            unmaskedIds=source_smallBatch.Left{word_tt};
            for ll=parameter.layer_num:-1:1
                if word_tt==T&&ll==parameter.layer_num
                    dh{ll}=d_source_sen(:,source_smallBatch.Sen);
                end
                if word_tt==1
                    c_t_1 =zeroState;
                else c_t_1=result.c_t_source_word{i}{ll,word_tt-1};
                end
                c_t=result.c_t_source_word{i}{ll,word_tt};
                lstm=result.lstms_source_word{i}{ll,word_tt};
                if word_tt==1
                    is_begin=1;
                else
                    is_begin=0;
                end
                [lstm_grad]=lstmUnitGrad(lstm,c_t, c_t_1,dc{ll},dh{ll},ll,word_tt,zeroState,parameter,parameter.Word_S{ll},is_begin);
                % backward calculation for lstm unit at source word level
                grad.Word_S{ll}=grad.Word_S{ll}+lstm_grad.W;
                lstm_grad.input(:,source_smallBatch.Delete{word_tt})=0;
                dh{ll}=lstm_grad.input(end-parameter.hidden+1:end,:);
                dc{ll}=lstm_grad.dc;
                if ll~=1
                    dh{ll-1}=dh{ll-1}+lstm_grad.input(1:parameter.hidden,:);
                else
                    % gradient for word tokens
                    embIndices=Word(unmaskedIds,word_tt)';
                    embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
                    numWords = length(embIndices);
                    allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
                    allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
                    wordCount = wordCount + numWords;
                end
            end
        end
    end
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = []; 
    [W_emb,indices] = aggregateMatrix(allEmbGrads, allEmbIndices);
    grad.vect(:,indices)=grad.vect(:,indices)+W_emb;
    clear allEmbIndices;
    clear allEmbGrads;
    clear W_emb;
    clear indices;
    clear result.c_t_source_word;
    clear result.h_t_source_word;
    clear result.lstms_source_word;
end


function[grad]=Backward_Target(current_batch,result,grad,parameter)
    % backward propagatin for target sentences
    num_sentence=size(current_batch.target_sen_matrix,1);
    for ll=1:parameter.layer_num
        grad.Word_T{ll}=zeroMatrix(size(parameter.Word_T{ll}));
        grad.Word_S{ll}=zeroMatrix(size(parameter.Word_S{ll}));
        grad.Sen_S{ll}=zeroMatrix(size(parameter.Sen_S{ll}));
        grad.Sen_T{ll}=zeroMatrix(size(parameter.Sen_T{ll}));
    end
    grad.vect=zeroMatrix(size(parameter.vect));

    T_target_sen=size(current_batch.target_sen_matrix,2);
    zeroState=zeroMatrix([parameter.hidden, size(current_batch.target_sen_matrix,1)]);
    L=[];

    for sen_tt=T_target_sen-1:-1:0
        target_sen=result.Target_sen{sen_tt+1};
        grad=DealSentenceTarget(current_batch,target_sen,sen_tt+1,grad,L,parameter,result);
        % back propagation for target sentences
        grad.vect(:,grad.indices)=grad.vect(:,grad.indices)+grad.W_emb;
        clear grad.indices;
        clear grad.W_emb;
        if sen_tt==0
            continue;
        end
        for ll=parameter.layer_num:-1:1
            if sen_tt==1
                dim=size(result.c_t_source_sen);
                c_t_1=result.c_t_source_sen{ll,dim(2)};
            else c_t_1=result.c_t_target_sen{ll,sen_tt-1};
            end
            c_t=result.c_t_target_sen{ll,sen_tt};
            lstm=result.lstms_target_sen{ll,sen_tt};
            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1,grad.target_sen_c{ll,sen_tt},grad.target_sen_h{ll,sen_tt},ll,sen_tt,zeroState,parameter,parameter.Sen_T{ll},0);
            % backpropagation for current lstm unit at sentence level
            lstm_grad.input(:,current_batch.target_delete{sen_tt+1})=0;
            lstm_grad.dc(:,current_batch.target_delete{sen_tt+1})=0;
            if sen_tt~=1
                grad.target_sen_h{ll,sen_tt-1}=lstm_grad.input(end-parameter.hidden+1:end,:)+grad.target_sen_h{ll,sen_tt-1};
                % if sentence index is not 1, gradient update for sentnence from previous time-step
                grad.target_sen_c{ll,sen_tt-1}=lstm_grad.dc+grad.target_sen_c{ll,sen_tt-1};
            else
                % else gradient update for vector representations from source sentences
                grad.source_h{ll,1}=grad.source_h{ll,1}+lstm_grad.input(end-parameter.hidden+1:end,:);
                grad.source_c{ll,1}=grad.source_c{ll,1}+lstm_grad.dc;
            end
            grad.Sen_T{ll}=grad.Sen_T{ll}+lstm_grad.W;
            if ll~=1
                grad.target_sen_h{ll-1,sen_tt}=lstm_grad.input(1:parameter.hidden,:)+grad.target_sen_h{ll-1,sen_tt};
            else
                L=lstm_grad.input(1:parameter.hidden,:);
            end
        end
    end
    for ll=1:parameter.layer_num
        grad.Word_T{ll}=grad.Word_T{ll}/num_sentence;
        grad.Sen_T{ll}=grad.Sen_T{ll}/num_sentence;
    end
    clear grad.target_sen_c;
    clear grad.target_sen_h;
    clear zeroState;
    clear lstm_grad;
    clear c_t_1;
    clear c_t;
    clear lstm;
    clear grad.ht;
    clear lstm_grad;
end

function[grad]=DealSentenceTarget(docbatch,target_sen,sen_tt,grad,end_grad,parameter,result)
    % back propagation to target sentences
    target_word=docbatch.target_word{sen_tt};
    Word=target_word.Word;
    Word_Delete=target_word.Delete;
    Word_Left=target_word.Left;
    N=size(Word,1);
    T=size(Word,2);
    zeroState=zeroMatrix([parameter.hidden,N]);
    for ll=parameter.layer_num:-1:1
        dh{ll} = zeroState;
        dc{ll} = zeroState;
    end
    wordCount = 0;
    numInputWords=size(Word,1)*size(Word,2);
    allEmbGrads=zeroMatrix([parameter.dimension,numInputWords]);
    allEmbIndices=[];
    for word_tt=T:-1:1
        if word_tt==T &&length(end_grad)==0
            continue;
        end
        unmaskedIds=Word_Left{word_tt};
        for ll=parameter.layer_num:-1:1
            if ll==parameter.layer_num
                if word_tt==T
                    if length(end_grad)~=0
                        dh{ll}=dh{ll}+end_grad;
                    end
                else 
                    dh{ll}=dh{ll}+grad.ht{sen_tt}{:,word_tt};
                end
            end
            if word_tt==1&& sen_tt==1
                dim=size(result.c_t_source_sen);
                c_t_1=result.c_t_source_sen{ll,dim(2)};
            elseif word_tt==1&& sen_tt~=1
                c_t_1=result.c_t_target_sen{ll,sen_tt-1};
            else c_t_1=target_sen.c_t_target_word{ll,word_tt-1};
            end
            c_t =target_sen.c_t_target_word{ll,word_tt};
            lstm =target_sen.lstms{ll,word_tt};
            %dh{ll}(:,Word_Delete{word_tt})=0;

            if length(Word_Delete{word_tt})~=0
                dh{ll}(:,Word_Delete{word_tt})=0;
                dc{ll}(:,Word_Delete{word_tt})=0;
            end
            % set values to 0

            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc{ll}, dh{ll},ll,word_tt,zeroState,parameter,parameter.Word_T{ll},0);
            % back propagation for current word level lstm unit
            grad.Word_T{ll}=grad.Word_T{ll}+lstm_grad.W;
            dc{ll} = lstm_grad.dc;
            dc{ll}(:,Word_Delete{word_tt})=0;
            dh{ll} = lstm_grad.input(end-parameter.hidden+1:end,:);
            if length(end_grad)~=0 && length(Word_Delete{word_tt})~=0&&ll==parameter.layer_num
                dh{ll}(:,Word_Delete{word_tt})=end_grad(:,Word_Delete{word_tt});
                % gradients for positions that have no words stay the same
            end


            if word_tt==1
                if sen_tt==1
                    grad.source_h{ll,1}=grad.source_h{ll,1}+dh{ll};
                    grad.source_c{ll,1}=grad.source_c{ll,1}+dc{ll};
                else
                    grad.target_sen_h{ll,sen_tt-1}=grad.target_sen_h{ll,sen_tt-1}+dh{ll};
                    grad.target_sen_c{ll,sen_tt-1}=grad.target_sen_c{ll,sen_tt-1}+dc{ll};
                end
            end
            if ll==1 
                % gradients for word embeddings
                embIndices=Word(unmaskedIds,word_tt)';
                embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
                numWords = length(embIndices);
                allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
                allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
                wordCount = wordCount + numWords;
            else
                dh{ll-1}(:,unmaskedIds)=dh{ll-1}(:,unmaskedIds)+lstm_grad.input(1:parameter.hidden,unmaskedIds);
            end
        end
    end
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
    [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices);
    clear dc;
    clear dh;
    clear allEmbGrads;
    clear allEmbIndices;
    clear allEmbGrads;
    clear lstm_grad;
end

function[lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc, dh, ll, t, zero_state,parameter,W,very_begin)
    % lstm gradient calculation
    dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
    do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, c_t, dh);
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);

    if very_begin==1
        df = zero_state;
    else 
        df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc);
    end
    lstm_grad.dc = lstm.f_gate.*dc;
    dl = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
    d_ifoa = [di; df; do; dl];
    lstm_grad.W = d_ifoa*lstm.input'; %dw
    lstm_grad.input=W'*d_ifoa;
    if parameter.dropout~=0
        if ll==1
            lstm_grad.input(1:parameter.dimension, :) = lstm_grad.input(1:parameter.dimension, :).*lstm.drop_left;
        else lstm_grad.input(1:parameter.hidden, :) = lstm_grad.input(1:parameter.hidden, :).*lstm.drop_left;
        end
    end
    if parameter.clip==1
        lstm_grad.input=arrayfun(@clipBackward, lstm_grad.input);
        lstm_grad.dc = arrayfun(@clipBackward, lstm_grad.dc);
    end
    clear dc; clear do; clear di; clear df; clear d_ifoa;
end


function [value] = plusTanhPrimeTriple(t, x, y, z)
    value = t + (1-x*x)*y*z;
end
function [value] = tanhPrimeTriple(x, y, z)
    value = (1-x*x)*y*z;
end
function [value] = plusMult(x, y, z)
    value = x + y*z;
end
function [value] = sigmoidPrimeTriple(x, y, z)
    value = x*(1-x)*y*z;
end

function [clippedValue] = clipBackward(x)
    if x>1000 clippedValue = single(1000);
    elseif x<-1000 clippedValue = single(-1000);
    else clippedValue =single(x);
    end
end
