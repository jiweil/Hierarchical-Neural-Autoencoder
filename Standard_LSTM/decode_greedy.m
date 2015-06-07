function[Translations]=decode_greedy(parameter,TestBatches,filename) 
% greedy decode
    disp('decode')
    Translations={};
    for batch_index=1:length(TestBatches)
        if mod(batch_index,20)==0
            batch_index
        end
        batch=TestBatches{batch_index};
        max_length=floor(batch.MaxLenSource*1.5);
        Word=batch.Word;
        N=size(Word,1);
        SourceLength=batch.SourceLength;
        [lstms,all_h_t,all_c_t]=Forward(batch,parameter,0);     % obtain vectors for source sentences
        last_h_t=all_h_t(:,size(Word,2));
        last_c_t=all_c_t(:,size(Word,2));
        first_words=BeamStep(parameter,last_h_t{parameter.layer_num},1);
        beamHistory=oneMatrix([N,max_length]);  % history
        beamHistory(:,1)=first_words(:);  % decode first words
        beamStates=cell(parameter.layer_num,1);
        for ll=1:parameter.layer_num
            beamStates{ll}.c_t=last_c_t{ll};
            beamStates{ll}.h_t=last_h_t{ll};
        end

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
                [beamStates{ll}, h_t, c_t]=lstmUnit(parameter.W_T{ll},parameter,x_t,h_t_1, c_t_1, ll, -1,0);    % lstmUnit, get current hidden vectors
                beamStates{ll}.h_t = h_t;
                beamStates{ll}.c_t = c_t;
            end
            all_next_words=BeamStep(parameter,beamStates{parameter.layer_num}.h_t,0);  
            % predict next words given current hidden time step vectors
            beamHistory(:,position+1)=all_next_words; 
            % add them to decoding history
        end
        for senId=1:N
            vector=beamHistory(senId,:);
            vector=vector(1:floor(1.5*batch.SourceLength(senId)));
            stop_sign=find(vector==parameter.stop);
            if length(stop_sign)==0
                dlmwrite(filename,vector,'delimiter',' ','-append');
            else
                dlmwrite(filename,vector(1:stop_sign-1),'delimiter',' ','-append');
            end
        end
    end
end

function[select_words]=BeamStep(parameter,h_t,isFirst) % next word prediction given hidden vectors
    if isFirst==1 scores=parameter.soft_W(1:end-1,:)*h_t;
    else scores=parameter.soft_W*h_t;
    end
    mx = max(scores);
    scores = bsxfun(@minus, scores, mx);
    logP=bsxfun(@minus, scores, log(sum(exp(scores))));
    [sortedLogP, sortedWords]=sort(logP, 'descend');
    select_words=sortedWords(1, :);

    probs_=exp(scores);
    norms = sum(probs_, 1);
    probs=bsxfun(@rdivide, probs_, norms);
end
