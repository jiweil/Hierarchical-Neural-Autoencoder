function[]=LSTM()
clear;
addpath('../misc');
n= gpuDeviceCount;
parameter.isGPU = 0;

if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(1);
else
    print('no gpu ! ! ! ! !');
end

parameter.dimension=1000;
parameter.alpha=0.1;    %learning rate
parameter.layer_num=4;  %number of layer
parameter.hidden=1000;
parameter.lstm_out_tanh=0;
parameter.Initial=0.08;
parameter.dropout=0.2;  %drop-out rate
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;  %whether check gradient or not.
parameter.PreTrainEmb=0;    %whether using pre-trained embeddings
parameter.update_embedding=1;   %whether update word embeddings
parameter.batch_size=32;    %mini-batch size
parameter.Source_Target_Same_Language=1;
%whether source and target is of the same language. For author-encoder task, it is.
parameter.maxGradNorm=1;    %gradient clipping
parameter.clip=1;

parameter.lr=5;
parameter.read=0;

if parameter.Source_Target_Same_Language==1
    parameter.Vocab=25001;  %vocabulary size plus document-end token
    parameter.stop=parameter.Vocab;     %document-end token
else
    parameter.SourceVocab=20;
    parameter.TargetVocab=20;
    parameter.stop=parameter.TargetVocab;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
end

if parameter.CheckGrad==1&parameter.dropout~=0  %use identical dropout-vector for gradient checking
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;


train_source_file='../data/train_permute.txt';
train_target_file='../data/train_permute.txt';
test_source_file='../data/test.txt';
test_target_file='../data/test.txt';


if 1==0
train_source_file='../data/very_small.txt';
train_target_file='../data/very_small.txt';
end

if 1==0
train_source_file='../toy.txt';
train_target_file='../toy.txt';
test_source_file='../toy.txt';
test_target_file='../toy.txt';
end
% above is files for gradient checking

if parameter.read==1
    disp('read');
    parameter=ReadParameter(parameter);     %read from exisitng parameter
else [parameter]=Initial(parameter);        %rand initialization
end

iter=0;

disp('begin')

while 1
    iter=iter+1;
    End=0;
    fd_train_source=fopen(train_source_file);   %read source
    fd_train_target=fopen(train_target_file);   %read target
    sum_cost=0;
    sum_num=0;
    batch_n=0;
    while 1
        batch_n=batch_n+1;
        [batch,End]=ReadTrainData(fd_train_source,fd_train_target,parameter);   %transform data to batches
        
        if End~=1 || (End==1&& length(batch.Word)~=0)   
            %not the end of document
            [lstm,all_h_t,c]=Forward(batch,parameter,1);    
            %LSTM Forward
            [batch_cost,grad]=softmax(all_h_t(parameter.layer_num,:),batch,parameter);      
            %softmax
            clear all_h_t;
            if (isnan(batch_cost)||isinf(batch_cost)) &&End~=1  
            %if gradient explodes
                if parameter.clip==1
                    fprintf('die !! Hopeless!!\n');
                    disp('read done');
                    parameter=pre_parameter;    
                    %load parameters stores from last step, and skip lately used batches
                    disp(batch_n)
                else parameter.clip=1;
                end
                if End==1 break;    
                %end of documents
                else continue;
                end
            end
            if parameter.isTraining==1
                grad=Backward(batch,grad,parameter,lstm,c);     
                %backward propagation
                disp('backward done')
                [parameter]=update_parameter(parameter,grad);   
                %update parameter
                clear lstm;
                clear c;
            end
        end
        if End==1
            fclose(fd_train_source);
            fclose(fd_train_target);
            break;
        end

        if mod(batch_n,500)==0  
            %save parameter every 500 batches
            pre_parameter=parameter;
        end
    end
    SaveParameter(parameter,iter);  
    %save parameter
end
end

function[parameter]=ReadParameter(parameter)    
    %read parameter
    for ll=1:parameter.layer_num
        W_file=strcat('save_parameter/_W_S',num2str(ll));
        parameter.W_S{ll}=gpuArray(load(W_file));
        W_file=strcat('save_parameter/_W_T',num2str(ll));
        parameter.W_T{ll}=gpuArray(load(W_file));
    end
    parameter.vect=gpuArray(load('save_parameter/_v'));
    parameter.soft_W=gpuArray(load('save_parameter/_soft_W'));
end

function SaveParameter(parameter,iter)
    if iter~=-1
        all=strcat('save_parameter/',int2str(iter));
        all=strcat(all,'_');
        all=strcat(all,num2str(parameter.alpha));
    else
        all='save_parameter/';
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_W_T');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.W_T{ll});
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_W_S');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.W_S{ll});
    end
    v_file=strcat(all,'_v');
    dlmwrite(v_file,parameter.vect);
    soft_W_file=strcat(all,'_soft_W');
    dlmwrite(soft_W_file,parameter.soft_W);
end

function[parameter]=update_parameter(parameter,grad)    
    %update parameters
    norm=computeGradNorm(grad,parameter);       
    %compute normalization
    if norm>parameter.maxGradNorm
        lr=parameter.alpha*parameter.maxGradNorm/norm;  
        %normalizing
    else lr=parameter.alpha;
    end
    for ll=1:parameter.layer_num
        parameter.W_T{ll}=parameter.W_T{ll}-lr*grad.W_T{ll};
        parameter.W_S{ll}=parameter.W_S{ll}-lr*grad.W_S{ll};
    end
    parameter.soft_W=parameter.soft_W-lr*grad.soft_W;
    parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-lr*grad.W_emb;
end

function[parameter]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        if i==1
            parameter.W_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
            parameter.W_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
        else
            parameter.W_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
            parameter.W_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
        end
    end
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,parameter.Vocab]);
    if parameter.Source_Target_Same_Language==1
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.Vocab,parameter.hidden]);
    else
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.TargetVocab,parameter.hidden]);
    end
end

function[current_batch,End]=ReadTrainData(fd_s,fd_t,parameter)
    tline_s = fgets(fd_s);
    tline_t = fgets(fd_t);
    i=0;
    Source={};Target={};
    End=0;
    while ischar(tline_s)
        i=i+1;
        text_s=deblank(tline_s);
        text_t=deblank(tline_t);
        if parameter.Source_Target_Same_Language~=1
            Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;  
            %reverse inputs
        else
            Source{i}=wrev(str2num(text_s));    
            %reverse inputs
        end
        Target{i}=[str2num(text_t),parameter.stop];     
        %add document_end_token
        if i==parameter.batch_size
            break;
        end
        tline_s = fgets(fd_s);
        tline_t = fgets(fd_t);
    end
    if ischar(tline_s)==0
        End=1;
    end
    current_batch=Batch();
    N=length(Source);
    for j=1:N
        source_length=length(Source{j});
        current_batch.SourceLength=[current_batch.SourceLength,source_length];
        if source_length>current_batch.MaxLenSource
            current_batch.MaxLenSource=source_length;
        end
        target_length=length(Target{j});
        if target_length>current_batch.MaxLenTarget
            current_batch.MaxLenTarget=target_length;
        end
    end
    total_length=current_batch.MaxLenSource+current_batch.MaxLenTarget;
    current_batch.MaxLen=total_length;
    current_batch.Word=ones(N,total_length);
    Mask=ones(N,total_length);
    % Mask: labeling positions where no words exisit. The purpose is to work on sentences in bulk making program faster
    for j=1:N
        source_length=length(Source{j});
        target_length=length(Target{j});
        current_batch.Word(j,current_batch.MaxLenSource-source_length+1:current_batch.MaxLenSource)=Source{j};      
        %words within sentences 
        current_batch.Word(j,current_batch.MaxLenSource+1:current_batch.MaxLenSource+target_length)=Target{j};
        Mask(j,1:current_batch.MaxLenSource-source_length)=0;       
        Mask(j,current_batch.MaxLenSource+target_length+1:end)=0;   
        % label positions without tokens 0
        current_batch.N_word=current_batch.N_word+target_length;
    end
    for j=1:total_length
        current_batch.Delete{j}=find(Mask(:,j)==0);
        current_batch.Left{j}=find(Mask(:,j)==1);
    end
    current_batch.Mask=Mask;
end

%gradient check
function check_soft_W(value1,i,j,batch,parameter)
    e=0.001;
    parameter.soft_W(i,j)=parameter.soft_W(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.soft_W(i,j)=parameter.soft_W(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_target_W(value1,ll,i,j,batch,parameter)
    e=0.001;
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function check_source_W(value1,ll,i,j,batch,parameter)
    e=0.001;
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function[norm]=computeGradNorm(grad,parameter)  %compute gradient norm
    norm=0;
    for ii=1:parameter.layer_num
        norm=norm+double(sum(grad.W_S{ii}(:).^2));
        norm=norm+double(sum(grad.W_T{ii}(:).^2));
    end
    norm=sqrt(norm);
end
