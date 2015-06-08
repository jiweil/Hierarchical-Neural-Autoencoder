function[Translations]=decode_greedy(parameter,batch,filename)
    T=size(batch.source_sen_matrix,2);
    % total number of time-steps
    N=size(batch.source_sen_matrix,1);
    % number of sentences
    L=length(batch.source_smallBatch);
    Max_Length=1.5*max(batch.num_word_in_doc);

    result=Forward(batch,parameter,0);
    % obtain embeddings for inputs
    first_words=BeamStep(parameter,result.source_sen{parameter.layer_num,1},1);
    % decode first words
    beamHistory=oneMatrix([N,Max_Length]);
    % store in history
    beamHistory(:,1)=first_words(:);

    for ll=1:parameter.layer_num
        beamSenStates_h{ll}=result.source_sen{ll,1};
        beamWordStates_h{ll}=result.source_sen{ll,1};
        beamSenStates_c{ll}=result.c_t_source_sen{ll,T};
        beamWordStates_c{ll}=result.c_t_source_sen{ll,T};
    end
    % update current lstm states
    for position=1:Max_Length
        words=beamHistory(:,position);
        % get earliest predicted words
        for ll=1:parameter.layer_num
            if ll==1 
                x_t=parameter.vect(:,words);
            else
                x_t=beamWordStates_h{ll-1};
            end
            h_t_1=beamWordStates_h{ll};
            c_t_1=beamWordStates_c{ll};
            [A,beamWordStates_h{ll},beamWordStates_c{ll}]=lstmUnit(parameter.Word_T{ll},parameter,x_t,h_t_1,c_t_1,ll,-1,0);
            % lstm unit calculation, get lsmt status for current time step at word level
        end
        end_sen_index=find(words==parameter.sen_stop);
        % if predicted words incode sentence-ending indicators
        if length(end_sen_index)~=0
            for ll=1:parameter.layer_num
                if ll==1
                    x_t=beamWordStates_h{ll}(:,end_sen_index);
                else
                    x_t=beamSenStates_h{ll-1}(:,end_sen_index);
                end
                h_t_1=beamSenStates_h{ll}(:,end_sen_index);
                c_t_1=beamSenStates_c{ll}(:,end_sen_index);
                [A,beamSenStates_h{ll}(:,end_sen_index),beamSenStates_c{ll}(:,end_sen_index)]=lstmUnit(parameter.Sen_T{ll},parameter,x_t,h_t_1,c_t_1,ll,-1,0);
                % update lsmt status for current time step at sentence level
                beamWordStates_h{ll}(:,end_sen_index)=beamSenStates_h{ll}(:,end_sen_index);
                beamWordStates_c{ll}(:,end_sen_index)=beamSenStates_c{ll}(:,end_sen_index);
            end
        end
        all_next_words=BeamStep(parameter,beamWordStates_h{parameter.layer_num},0);
        % predict next word
        beamHistory(:,position+1)=all_next_words;
    end
    for senId=1:N
        vector=beamHistory(senId,:);
        vector=vector(1:floor(1.5*batch.num_word_in_doc(senId)));
        % maximium length of 1.5 * length of orginal inputs
        stop_sign=find(vector==parameter.doc_stop);
        if length(stop_sign)==0
            vector=vector;
        else
            vector=vector(1:stop_sign-1);
        end
        M=find(vector==parameter.sen_stop);
        vector(M)=[];
        % remove sentence-ending token
        dlmwrite(filename,vector,'delimiter',' ','-append');
    end
end

function[select_words]=BeamStep(parameter,h_t,isFirst)
    if isFirst==1 scores=parameter.soft_W(1:end-2,:)*h_t;
    else scores=parameter.soft_W*h_t;
    end
    % could not predict sentence-ending token or document-ending token for first word
    mx = max(scores);
    scores = bsxfun(@minus, scores, mx);
    logP=bsxfun(@minus, scores, log(sum(exp(scores))));
    [sortedLogP, sortedWords]=sort(logP, 'descend');
    select_words=sortedWords(1, :);

    probs_=exp(scores);
    norms = sum(probs_, 1);
    probs=bsxfun(@rdivide, probs_, norms);
end
