function[result]=Forward(docbatch,parameter,isTraining)
    [result]=Forward_Source_Word(docbatch,parameter,isTraining);
    % forward calculation at word level for each sentence
    [result]=Forward_Source_Sen(result,docbatch,parameter,isTraining);
    % forward calculation at sentence level for each document
    if isTraining==1
        [result]=Forward_Target(result,docbatch,parameter,isTraining);
    end
end

function[result]=Forward_Source_Word(docbatch,parameter,isTraining)
    % forward calculation at word level for each sentence
    sourceBatch=docbatch.source_smallBatch;
    result.source_sen_vector=[];
    for i=1:length(sourceBatch)
        batch=sourceBatch{i};
        T=batch.max_length;
        h_t_source_word=cell(parameter.layer_num,T);
        % store h_t
        result.c_t_source_word{i}=cell(parameter.layer_num,T);
        % store c_t
        result.lstms_source_word{i} = cell(parameter.layer_num,T);
        % store gate values for lstm units 
        N=size(batch.Word,1);
        zeroState=zeroMatrix([parameter.hidden,N]);
        for ll=1:parameter.layer_num
            for tt=1:T
                h_t_source_word{ll,tt}=zeroState;
                result.c_t_source_word{i}{ll,tt}=zeroState;
            end
        end
        for t=1:T
            for ll=1:parameter.layer_num
                W=parameter.Word_S{ll};
                % W for word level composition in source
                if t==1
                    h_t_1=zeroState;
                    c_t_1=zeroState;
                else
                    c_t_1=result.c_t_source_word{i}{ll, t-1};
                    h_t_1=h_t_source_word{ll, t-1};
                end
                if ll==1
                    x_t=parameter.vect(:,batch.Word(:,t));
                else
                    x_t=h_t_source_word{ll-1,t};
                end
                x_t(:,batch.Delete{t})=0;
                h_t_1(:,batch.Delete{t})=0;
                c_t_1(:,batch.Delete{t})=0;
                % set postion that do not have words to zero
                [result.lstms_source_word{i}{ll, t},h_t_source_word{ll, t},result.c_t_source_word{i}{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
                % compute lstm unit
                if t==T && ll==parameter.layer_num
                    result.source_sen_vector=[result.source_sen_vector,h_t_source_word{ll,t}];
                    % store vector embeddings for each source sentence
                    clear h_t_source_word;
                end
            end
        end
    end
    clear x_t;
    clear h_t_1;
    clear c_t_1;
end

function[result]=Forward_Source_Sen(result,docbatch,parameter,isTraining)
    % forward calculation at sentence level for each document
    T=docbatch.max_source_sen;
    h_t_source_sen=cell(parameter.layer_num,T);
    result.c_t_source_sen=cell(parameter.layer_num,T);
    result.lstms_source_sen=cell(parameter.layer_num,T);
    result.source_sen=cell(parameter.layer_num,1);

    N=size(docbatch.source_sen_matrix,1);
    zeroState=zeroMatrix([parameter.hidden,N]);
    for ll=1:parameter.layer_num
        for tt=1:T
            h_t_source_sen{ll,tt}=zeroState;
            result.c_t_source_sen{ll,tt}=zeroState;
        end
    end
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.Sen_S{ll};
            % sentence-level compostion
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 =result.c_t_source_sen{ll, t-1};
                h_t_1 =h_t_source_sen{ll, t-1};
            end
            if ll==1
                x_t=result.source_sen_vector(:,docbatch.source_sen_matrix(:,t));
                % sentence-level embedding
            else
                x_t=h_t_source_sen{ll-1,t};
            end
            x_t(:,docbatch.source_delete{t})=0;
            h_t_1(:,docbatch.source_delete{t})=0;
            c_t_1(:,docbatch.source_delete{t})=0;
            % set values for deleted postion to 0
            [result.lstms_source_sen{ll, t},h_t_source_sen{ll, t},result.c_t_source_sen{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
            % lstm unit calculation at sentence level
            if t==T 
                result.source_sen{ll,1}=h_t_source_sen{ll, t};
                % store vector embeddings for documents
            end
        end
    end
    clear result.source_sen_vector;
    clear h_t_source_sen;
    clear x_t;
    clear h_t_1;
    clear c_t_1;
end

function[result]=Forward_Target(result,docbatch,parameter,isTraining)
    % forward for target documents
    T=docbatch.max_target_sen;
    result.h_t_target_sen=cell(parameter.layer_num,T);
    % sentence level
    result.c_t_target_sen=cell(parameter.layer_num,T);
    result.lstms_target_sen=cell(parameter.layer_num,T);
    N=size(docbatch.target_sen_matrix,1);
    zeroState=zeroMatrix([parameter.hidden,N]);

    for ll=1:parameter.layer_num
        for tt=1:T
            result.h_t_target_sen{ll,tt}=zeroState;
            result.c_t_target_sen{ll,tt}=zeroState;
        end
    end
    result.Target_sen={};
    for sen_tt=1:T
        for ll=1:parameter.layer_num
            W=parameter.Sen_T{ll};
            % sentnece compositions for target
            if sen_tt==1
                % if sentence index is 1, h_t_1 and c_t_1 are from outputs from source sentences
                h_t_1=result.source_sen{ll,1};
                dim=size(result.c_t_source_sen);
                c_t_1=result.c_t_source_sen{ll,dim(2)};
            else
                c_t_1 =result.c_t_target_sen{ll, sen_tt-1};
                % otherwise, c_t_1 are lstm outputs from last time step
                h_t_1 =result.h_t_target_sen{ll, sen_tt-1};
            end
            if ll==1
                Word_List=docbatch.target_word{sen_tt}.Word;
                Word_Delete=docbatch.target_word{sen_tt}.Delete;
                if sen_tt==1 
                    M1=result.source_sen(:,1);
                    dim=size(result.c_t_source_sen);
                    M2=result.c_t_source_sen(:,dim(2));
                else
                    M1=result.h_t_target_sen(:,sen_tt-1);
                    M2=result.c_t_target_sen(:,sen_tt-1);
                end
                result.Target_sen{sen_tt}=Forward_Target_Word(M1,M2,Word_List,Word_Delete,docbatch,parameter,isTraining);
                % if layer number is 1, x_t sentence level embeddings based on its containing words; compute sentence level embeddings
                x_t=result.Target_sen{sen_tt}.h_t_target_word{parameter.layer_num,size(Word_List,2)};
            else
                x_t=result.h_t_target_sen{ll-1,sen_tt};
                % else, x_t is outputs from last layer lstm
            end
            x_t(:,docbatch.target_delete{sen_tt})=0;
            h_t_1(:,docbatch.target_delete{sen_tt})=0;
            c_t_1(:,docbatch.target_delete{sen_tt})=0;
            % set deleted postions to 0 value
            [result.lstms_target_sen{ll,sen_tt},result.h_t_target_sen{ll,sen_tt},result.c_t_target_sen{ll,sen_tt}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,sen_tt,isTraining);
            % lstm unit calculation
        end
    end
    clear target_sen_vector;
    clear x_t;
    clear h_t_1;
    clear c_t_1;
end

function[target_sen]=Forward_Target_Word(h_t_sen,c_t_sen,Word_List,Word_Delete,docbatch,parameter,isTraining)
    % obtain sentence level embeddings for target sentences
    N=size(Word_List,1);
    T=size(Word_List,2);
    target_sen.h_t_target_word=cell(parameter.layer_num,T);
    target_sen.c_t_target_word=cell(parameter.layer_num,T);
    target_sen.lstms=cell(parameter.layer_num,T);
    zeroState=zeroMatrix([parameter.hidden,N]);

    for ll=1:parameter.layer_num
        for tt=1:T
            target_sen.h_t_target_word{ll,tt}=zeroState;
            target_sen.c_t_target_word{ll,tt}=zeroState;
        end
    end
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.Word_T{ll};
            if t==1
                h_t_1=h_t_sen{ll,1};
                c_t_1=c_t_sen{ll,1};
            else
                c_t_1 =target_sen.c_t_target_word{ll, t-1};
                h_t_1 =target_sen.h_t_target_word{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,Word_List(:,t));
                % if ll=1, x_t are correspondent word embeddings
            else
                x_t=target_sen.h_t_target_word{ll-1,t};
                % else x_t are outputs from previous layer
            end
            x_t(:,Word_Delete{t})=0;
            h_t_1(:,Word_Delete{t})=0;
            c_t_1(:,Word_Delete{t})=0;
            % set deleted positions to 0
            [target_sen.lstms{ll, t},target_sen.h_t_target_word{ll, t},target_sen.c_t_target_word{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
            % lstm unit calculations 
            if t~=1
                target_sen.h_t_target_word{ll, t}(:,Word_Delete{t})=target_sen.h_t_target_word{ll,t-1}(:,Word_Delete{t});
                % sentence representations stay the same for deleted positions
            end
        end
    end
    clear h_t_1;
    clear c_t_1;
    clear x_t;
end
