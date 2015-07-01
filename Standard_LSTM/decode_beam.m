function[]=decode_beam(parameter,TestBatches,filename)
    disp('decode')
    parameter.beamSize=7;
    for batch_index=1:length(TestBatches)
        batch_index
        batch=TestBatches{batch_index};
        max_length=floor(batch.MaxLenSource*1);
        Word=batch.Word;
        N=size(Word,1);
        SourceLength=batch.SourceLength;
        [lstms,last_h_t,last_c_t]=Forward(batch,parameter,0);
        numElements =N*parameter.beamSize;
        translations=-zeroMatrix([N,max_length]);
        [first_scores,first_words]=BeamStep(parameter,last_h_t{parameter.layer_num},1);
        beamHistory=oneMatrix([numElements,max_length]);
        beamHistory(:,1)=first_words(:);
        beamScores=first_scores(:)';
        beamStates=cell(parameter.layer_num,1);
        for ll=1:parameter.layer_num
            beamStates{ll}.c_t=reshape(repmat(last_c_t{ll},parameter.beamSize,1),parameter.hidden,numElements);
            beamStates{ll}.h_t=reshape(repmat(last_h_t{ll},parameter.beamSize,1),parameter.hidden,numElements);
        end
        decoded=zeros(1,N);
        INDECE=oneMatrix([1,numElements]);
        for position=1:max_length
            words=beamHistory(:,position);
            for ll=1:parameter.layer_num
                if ll == 1
                    x_t=parameter.vect(:,words);
                else
                    x_t=beamStates{ll-1}.h_t;
                end
                h_t_1 = beamStates{ll}.h_t;
                c_t_1 = beamStates{ll}.c_t;
                [beamStates{ll}, h_t, c_t]=lstmUnit(parameter.W_T{ll},parameter,x_t,h_t_1, c_t_1, ll, -1,0);
                beamStates{ll}.h_t = h_t;
                beamStates{ll}.c_t = c_t;
            end
            [all_next_scores,all_next_words]=BeamStep(parameter,beamStates{parameter.layer_num}.h_t,0);
            all_next_scores=bsxfun(@plus,all_next_scores,beamScores);
            all_next_scores=reshape(all_next_scores,[parameter.beamSize*parameter.beamSize,N]);
            all_next_words=reshape(all_next_words,[parameter.beamSize*parameter.beamSize,N]);
            [all_next_scores,indices]=sort(all_next_scores,'descend');
            for sentId=1:N
                if decoded(sentId)==1 continue; end
                sorted_Indices = indices(:, sentId);
                sorted_next_words=all_next_words(sorted_Indices,sentId);
                end_index=find(sorted_next_words(1:parameter.beamSize)==parameter.stop,1);
                if sorted_next_words(1)==parameter.stop&&position>floor(SourceLength(sentId)/2)
                    decoded(sentId)=1;
                    end_index=sorted_Indices(1);
                    previous_index=floor((end_index-1)/parameter.beamSize)+1;
                    translations(sentId,1:position+1)=[beamHistory((sentId-1)*parameter.beamSize+previous_index,1:position),parameter.stop];
                    continue;
                end
                if position==max_length
                    end_index=sorted_Indices(1);
                    previous_index=floor((end_index-1)/parameter.beamSize)+1;
                    translations(sentId,1:position+1)=[beamHistory((sentId-1)*parameter.beamSize+previous_index,1:position),sorted_next_words(1)];
                    continue;
                end
                next_word_index=find(sorted_Indices~=parameter.stop,parameter.beamSize);
                beamScores(parameter.beamSize*(sentId-1)+1:parameter.beamSize*sentId)=all_next_scores(next_word_index,sentId);
                previous_index=(sentId-1)*parameter.beamSize+ floor((sorted_Indices(next_word_index)-1)/parameter.beamSize)+1;
                T=(sentId-1)*parameter.beamSize+1:(sentId-1)*parameter.beamSize+parameter.beamSize;
                beamHistory(T,:)=beamHistory(previous_index,:);
                beamHistory(T,position+1)=sorted_next_words(1:parameter.beamSize);
                INDECE(T)=previous_index;
            end
            if sum(decoded)==N
                break;
            end
            for ll=1:parameter.layer_num
                beamStates{ll}.c_t=beamStates{ll}.c_t(:,INDECE);
                beamStates{ll}.h_t=beamStates{ll}.h_t(:,INDECE);
            end
        end
        
        for senId=1:N
            vector=translations(senId,:);
            vector=vector(1:floor(1*batch.SourceLength(senId)));
            stop_sign=find(vector==parameter.stop);
            if length(stop_sign)==0
                dlmwrite(filename,vector,'delimiter',' ','-append');
            else
                dlmwrite(filename,vector(1:stop_sign-1),'delimiter',' ','-append');
            end
        end
    end
end

function[select_logP,select_words]=BeamStep(parameter,h_t,isFirst)
    if isFirst==1 scores=parameter.soft_W(2:end-1,:)*h_t;
    else scores=parameter.soft_W(2:end,:)*h_t;
    end
    mx = max(scores);
    scores = bsxfun(@minus, scores, mx);
    logP=bsxfun(@minus, scores, log(sum(exp(scores))));
    [sortedLogP, sortedWords]=sort(logP, 'descend');
    select_words=1+sortedWords(1:parameter.beamSize, :);
    select_logP=sortedLogP(1:parameter.beamSize, :);
    probs=exp(scores);
    norms = sum(probs, 1);
    probs=bsxfun(@rdivide, probs, norms);
end
