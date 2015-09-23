function[batch,Stop]=ReadData(fd_s,fd_t,parameter)
    tline_s = fgets(fd_s);
    Target={};
    Stop=0;
    doc_index=1;
    sen_index=0;
    i=0;

    length_array=[];
    num_of_sen=0;
    max_source_sen=0;
    while ischar(tline_s)
        text_s=deblank(tline_s);
        if length(text_s)==0
            sourceDoc{doc_index}=[i-sen_index+1,i];
            doc_index=doc_index+1;
            if sen_index>max_source_sen
                max_source_sen=sen_index;
            end
            sen_index=0;
            if doc_index==parameter.batch_size+1
                break;
            end
        else
            i=i+1;
            sen_index=sen_index+1;
            num_of_sen=num_of_sen+1;
            if parameter.Source_Target_Same_Language~=1
                Source{i}=str2num(text_s)+parameter.TargetVocab;
            else
                Source{i}=str2num(text_s);
            end
            length_array=[length_array;length(Source{i})];
        end
        tline_s = fgets(fd_s);
    end
    if ischar(tline_s)==0
        Stop=1;
    end
    if exist('sourceDoc')==0
        batch.source_smallBatch=[];
        return;
    end
    batch.num_of_source_sen=num_of_sen;
    source_sen_matrix=ones(length(sourceDoc),max_source_sen);
    source_sen_mask=ones(length(sourceDoc),max_source_sen);
    num_word_in_doc=[];
    for i=1:length(sourceDoc)
        L=sourceDoc{i}(1):sourceDoc{i}(2);
        n_word=0;
        for j=sourceDoc{i}(1):sourceDoc{i}(2)
            n_word=n_word+length(Source{j});
        end
        num_word_in_doc=[num_word_in_doc,n_word];
        l=length(L);
        source_sen_matrix(i,max_source_sen-l+1:max_source_sen)=L;
        source_sen_mask(i,1:max_source_sen-l)=0;
    end
    batch.num_word_in_doc=num_word_in_doc;
    source_delete={};
    for j=1:size(source_sen_mask,2)
        source_delete{j}=find(source_sen_mask(:,j)==0);
        source_left{j}=find(source_sen_mask(:,j)==1);
    end
    [a,b]=sort(length_array);
    c=ones(length(b),1);
    c(b)=1:length(b);
    source_sen_matrix=c(source_sen_matrix);
    if size(source_sen_matrix,2)==1
        source_sen_matrix=source_sen_matrix';
    end

    source_smallBatch={};
    sen_size=32;
    num_small=ceil(num_of_sen/sen_size);
    for i=1:num_small
        Begin=sen_size*(i-1)+1;
        End=sen_size*i;
        if End>num_of_sen
            End=num_of_sen;
        end
        max_length=-1;
        for j=Begin:End
            if length(Source{b(j)})>max_length
                max_length=length(Source{b(j)});
            end
        end
        N=End-Begin+1;
        source_smallBatch{i}.max_length=max_length;
        source_smallBatch{i}.Word=ones(N,max_length);
        source_smallBatch{i}.Mask=ones(N,max_length);
        source_smallBatch{i}.Sen=Begin:End;
        for j=Begin:End
            source_length=length(Source{b(j)});
            source_smallBatch{i}.Word(j-Begin+1,max_length-source_length+1:max_length)=Source{b(j)};
            source_smallBatch{i}.Mask(j-Begin+1,1:max_length-source_length)=0;
        end
        for j=1:max_length
            source_smallBatch{i}.Delete{j}=find(source_smallBatch{i}.Mask(:,j)==0);
            source_smallBatch{i}.Left{j}=find(source_smallBatch{i}.Mask(:,j)==1);
        end
        %disp('source')
        %source_smallBatch{i}.Word
        %source_smallBatch{i}.Mask
    end
    

    End=0;
    doc_index=1;
    sen_index=0;
    i=0;
    Target={};
    target_doc_index_array=[];
    target_sen_index_array=[];
    max_target_sen=0;
    length_array=[];
    num_of_sen=0;
    tline_t = fgets(fd_t);
    targetDoc={};
    while ischar(tline_t)
        text_t=deblank(tline_t);
        if length(text_t)==0
            Target{i}(length(Target{i}))=parameter.doc_stop;
            targetDoc{doc_index}=[i-sen_index+1,i];
            doc_index=doc_index+1;
            if sen_index>max_target_sen
                max_target_sen=sen_index;
            end
            sen_index=0;
            if doc_index==parameter.batch_size+1
                break;
            end
        else
            i=i+1;
            num_of_sen=num_of_sen+1;
            sen_index=sen_index+1;
            Target{i}=[str2num(text_t),parameter.sen_stop];
            length_array=[length_array;length(Target{i})];
        end
        tline_t = fgets(fd_t);
    end

    target_sen_matrix=ones(length(targetDoc),max_target_sen);
    target_sen_mask=ones(length(targetDoc),max_target_sen);
    for i=1:length(targetDoc)
        L=targetDoc{i}(1):targetDoc{i}(2);
        l=length(L);
        target_sen_matrix(i,1:l)=L;
        target_sen_mask(i,l+1:max_target_sen)=0;
    end
    target_word={};
    for i=1:size(target_sen_matrix,2)
        Max_l=1;
        sen_list=target_sen_matrix(:,i);
        mask_list=target_sen_mask(:,i);
        for j=1:length(sen_list)
            if mask_list(j)==0
                continue;
            end
            index=sen_list(j);
            Length=length(Target{index});
            if Length>Max_l
                Max_l=Length;
            end
        end
        target_word{i}.Word=ones(length(sen_list),Max_l);
        target_word{i}.Mask=ones(length(sen_list),Max_l);
        target_word{i}.End=[];
        for j=1:length(sen_list)
            if mask_list(j)==0
                target_word{i}.Mask(j,:)=0;
                continue;
            end
            index=sen_list(j);
            Length=length(Target{index});
            target_word{i}.Word(j,1:Length)=Target{index};
            target_word{i}.Mask(j,Length+1:end)=0;
        end
        for j=1:size(target_word{i}.Mask,2)
            target_word{i}.Delete{j}=find(target_word{i}.Mask(:,j)==0);
            target_word{i}.Left{j}=find(target_word{i}.Mask(:,j)==1);
        end
    end
    target_delete={};
    for j=1:size(target_sen_mask,2)
        target_delete{j}=find(target_sen_mask(:,j)==0);
    end

    batch.source_smallBatch=source_smallBatch;
    batch.max_source_sen=max_source_sen;
    batch.max_target_sen=max_target_sen;

    batch.target_sen_matrix=target_sen_matrix;
    batch.source_sen_matrix=source_sen_matrix;
    batch.target_sen_mask=target_sen_mask;
    batch.source_sen_mask=source_sen_mask;
    batch.source_delete=source_delete;
    batch.source_left=source_left;

    batch.target_delete=target_delete;
    batch.target_word=target_word;

    
    clear target_sen_matrix;
    clear source_sen_matrix;
    clear target_sen_mask;
    clear source_sen_mask;
    clear source_smallBatch;
    clear target_doc_index_array;

    clear source_delete;
    clear source_left;
    clear target_delete;

    clear target_word;
end
