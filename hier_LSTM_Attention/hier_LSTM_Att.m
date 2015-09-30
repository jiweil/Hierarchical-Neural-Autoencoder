function[]=hier_LSTM_Att()
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
parameter.dropout=0;
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=0;
%whether using pre-trained embeddings
parameter.update_embedding=1;
%whether update word embeddings
parameter.batch_size=16;
parameter.Source_Target_Same_Language=1;
%whether source and target is of the same language. For author-encoder task, it is.
parameter.maxGradNorm=1;
parameter.clip=0;

parameter.lr=5;
parameter.read=0;

if parameter.Source_Target_Same_Language==1
    parameter.Vocab=25002;
    %vocabulary size plus sentence-end and document-end
    parameter.sen_stop=parameter.Vocab-1;   %sentence-end
    parameter.doc_stop=parameter.Vocab;     %document-end
else
    parameter.SourceVocab=20;
    parameter.TargetVocab=20;
    parameter.stop=parameter.TargetVocab;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
end

if parameter.CheckGrad==1&parameter.dropout~=0
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end
%alpha: learning rate for minibatch

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;


train_source_file='data/train_source_permute_segment.txt';
train_target_file='data/train_target_permute_segment.txt';

if 1==0
disp('small data testing')
train_source_file='small_data/train_source_permute_segment_small.txt';
train_target_file='small_data/train_target_permute_segment_small.txt';
end


if parameter.read==1
    disp('read');
    parameter=ReadParameter(parameter);
    % read parameters
else [parameter]=Initial(parameter);
end

pre_parameter=parameter;

iter=0;

disp('begin')

while 1
    iter=iter+1
    fd_train_source=fopen(train_source_file);
    fd_train_target=fopen(train_target_file);
    sum_cost=0;
    sum_num=0;
    batch_n=0;
    while 1
        batch_n=batch_n+1;
        [current_batch,End]=ReadData(fd_train_source,fd_train_target,parameter);
        %   read one batch
        if End~=1 || (End==1&& length(current_batch.source_smallBatch)~=0)
            result=Forward(current_batch,parameter,1);  
            % Forward
            [batch_cost,grad]=softmax(result,current_batch,parameter);
            if (isnan(batch_cost)||isinf(batch_cost)) &&End~=1
                % if batch_cost is nan, load ealier parameters and skip latest batches
                if parameter.clip==1
                    fprintf('die !! Hopeless!!\n');
                    disp('read done');
                    parameter=pre_parameter;
                    disp(batch_n)
                else parameter.clip=1;
                end
                if End==1 break;
                else continue;
                end
            end
            if parameter.isTraining==1
                grad=Backward(current_batch,result,grad,parameter);
                % backward propagation
                clear result;
                if 1==0
                    check(grad,current_batch,parameter)
                    % check gradient
                end
                [parameter]=update_parameter(parameter,grad);
                % update parameter
                clear lstm;
                clear c;
            end
        end
        if End==1
            fclose(fd_train_source);
            fclose(fd_train_target);
            break;
        end

        if mod(batch_n,100)==0
            pre_parameter=parameter;
            % store parameters from last 100 batches in case of gradient explosion
        end
    end

    SaveParameter(parameter,iter);
end
end
function[]=check(grad,current_batch,parameter)
    check_Attention_U(grad.Attention_U(1),1,current_batch,parameter);
    check_Attention_W(grad.Attention_W(1,1),1,1,current_batch,parameter);
    if 1==0
    check_vect(grad.vect(1,1),1,1,current_batch,parameter);
    check_vect(grad.vect(1,2),1,2,current_batch,parameter);
    check_vect(grad.vect(1,3),1,3,current_batch,parameter);
    check_vect(grad.vect(1,4),1,4,current_batch,parameter);
    check_vect(grad.vect(1,5),1,5,current_batch,parameter);
    end
    if 1==0
    disp('word_T')
    check_word_T(grad.Word_T{1}(1,1),1,1,1,current_batch,parameter);
    check_word_T(grad.Word_T{2}(1,1),2,1,1,current_batch,parameter);
    check_word_T(grad.Word_T{3}(1,1),3,1,1,current_batch,parameter);
    check_word_T(grad.Word_T{4}(1,1),4,1,1,current_batch,parameter);
    end
    if 1==1
    disp('sen_T')
    check_sen_T(grad.Sen_T{1}(1,1),1,1,1,current_batch,parameter);
    check_sen_T(grad.Sen_T{2}(1,1),2,1,1,current_batch,parameter);
    check_sen_T(grad.Sen_T{3}(1,1),3,1,1,current_batch,parameter);
    check_sen_T(grad.Sen_T{4}(1,1),4,1,1,current_batch,parameter);
    end
    if 1==1
    disp('sen_S')
    check_sen_S(grad.Sen_S{1}(1,1),1,1,1,current_batch,parameter);
    check_sen_S(grad.Sen_S{1}(1,2),1,1,2,current_batch,parameter);
    check_sen_S(grad.Sen_S{1}(1,7),1,1,7,current_batch,parameter);
    check_sen_S(grad.Sen_S{1}(1,8),1,1,8,current_batch,parameter);
    end
    if 1==1
    disp('word_S')
    check_word_S(grad.Word_S{1}(1,1),1,1,1,current_batch,parameter);
    check_word_S(grad.Word_S{1}(1,2),1,1,2,current_batch,parameter);
    check_word_S(grad.Word_S{2}(1,1),2,1,1,current_batch,parameter);
    check_word_S(grad.Word_S{2}(1,2),2,1,2,current_batch,parameter);
    check_word_S(grad.Word_S{3}(1,1),3,1,1,current_batch,parameter);
    check_word_S(grad.Word_S{3}(1,2),3,1,2,current_batch,parameter);
    end
end

function[parameter]=ReadParameter(parameter)
    % read parameters
    for ll=1:parameter.layer_num
        W_file=strcat('save_parameter/_Word_S',num2str(ll));
        parameter.Word_S{ll}=gpuArray(load(W_file));
        W_file=strcat('save_parameter/_Word_T',num2str(ll));
        parameter.Word_T{ll}=gpuArray(load(W_file));
        W_file=strcat('save_parameter/_Sen_S',num2str(ll));
        parameter.Sen_S{ll}=gpuArray(load(W_file));
        W_file=strcat('save_parameter/_Sen_T',num2str(ll));
        parameter.Sen_T{ll}=gpuArray(load(W_file));
    end
    parameter.Attention_W=gpuArray(load('save_parameter/_Attention_W'));
    parameter.Attention_U=gpuArray(load('save_parameter/_Attention_U'));
    parameter.vect=gpuArray(load('save_parameter/_v'));
    parameter.soft_W=gpuArray(load('save_parameter/_soft_W'));
end

function SaveParameter(parameter,iter)
    % save parameters
    if iter~=-1
        all=strcat('save_parameter/',int2str(iter));
        all=strcat(all,'_');
        all=strcat(all,num2str(parameter.alpha));
    else
        all='save_parameter/';
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_Word_T');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.Word_T{ll});
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_Word_S');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.Word_S{ll});
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_Sen_T');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.Sen_T{ll});
    end
    for ll=1:parameter.layer_num
        W_file=strcat(all,'_Sen_S');
        W_file=strcat(W_file,num2str(ll));
        dlmwrite(W_file,parameter.Sen_S{ll});
    end
    W_file=strcat(all,'_Attention_W');
    dlmwrite(W_file,parameter.Attention_W)
    U_file=strcat(all,'_Attention_U');
    dlmwrite(U_file,parameter.Attention_U)

    v_file=strcat(all,'_v');
    dlmwrite(v_file,parameter.vect);
    soft_W_file=strcat(all,'_soft_W');
    dlmwrite(soft_W_file,parameter.soft_W);
end

function[parameter]=update_parameter(parameter,grad)
    % update parameters
    norm=computeGradNorm(grad,parameter);
    % compute norm
    if norm>parameter.maxGradNorm
        lr=parameter.alpha*parameter.maxGradNorm/norm;
    else lr=parameter.alpha;
    end
    for ll=1:parameter.layer_num
        parameter.Word_S{ll}=parameter.Word_S{ll}-lr*grad.Word_S{ll};
        parameter.Word_T{ll}=parameter.Word_T{ll}-lr*grad.Word_T{ll};
        parameter.Sen_S{ll}=parameter.Sen_S{ll}-lr*grad.Sen_S{ll};
        parameter.Sen_T{ll}=parameter.Sen_T{ll}-lr*grad.Sen_T{ll};
    end
    parameter.Attention_U=parameter.Attention_U-lr*grad.Attention_U;
    parameter.Attention_W=parameter.Attention_W-lr*grad.Attention_W;
    parameter.soft_W=parameter.soft_W-lr*grad.soft_W;
    parameter.vect=parameter.vect-lr*grad.vect;
    clear grad;
end

function[parameter]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        if i==1
            parameter.Word_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
            parameter.Word_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
        else
            parameter.Word_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
            parameter.Word_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
        end
        parameter.Sen_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
        if i==1
            parameter.Sen_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,3*parameter.hidden]);
        else
            parameter.Sen_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
        end
            
    end
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,parameter.Vocab]);
    if parameter.Source_Target_Same_Language==1
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.Vocab,parameter.hidden]);
    else
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.TargetVocab,parameter.hidden]);
    end
    parameter.Attention_W=randomMatrix(parameter.Initial,[parameter.hidden,parameter.hidden*2]);
    parameter.Attention_U=randomMatrix(parameter.Initial,[parameter.hidden,1]);
end

function check_Attention_W(value1,i,j,current_batch,parameter)
    e=0.001;
    parameter.Attention_W(i,j)=parameter.Attention_W(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Attention_W(i,j)=parameter.Attention_W(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Attention_W(i,j)=parameter.Attention_W(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_Attention_U(value1,i,current_batch,parameter)
    e=0.001;
    parameter.Attention_U(i)=parameter.Attention_U(i)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Attention_U(i)=parameter.Attention_U(i)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Attention_U(i)=parameter.Attention_U(i)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_vect(value1,i,j,current_batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_soft_W(value1,i,j,current_batch,parameter)
    e=0.001;
    parameter.soft_W(i,j)=parameter.soft_W(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.soft_W(i,j)=parameter.soft_W(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.soft_W(i,j)=parameter.soft_W(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_word_T(value1,ll,i,j,current_batch,parameter)
    e=1e-3;
    parameter.Word_T{ll}(i,j)=parameter.Word_T{ll}(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Word_T{ll}(i,j)=parameter.Word_T{ll}(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Word_T{ll}(i,j)=parameter.Word_T{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_word_S(value1,ll,i,j,current_batch,parameter)
    e=0.001;
    parameter.Word_S{ll}(i,j)=parameter.Word_S{ll}(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Word_S{ll}(i,j)=parameter.Word_S{ll}(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Word_S{ll}(i,j)=parameter.Word_S{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end


function check_sen_S(value1,ll,i,j,current_batch,parameter)
    e=0.001;
    parameter.Sen_S{ll}(i,j)=parameter.Sen_S{ll}(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Sen_S{ll}(i,j)=parameter.Sen_S{ll}(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Sen_S{ll}(i,j)=parameter.Sen_S{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_sen_T(value1,ll,i,j,current_batch,parameter)
    e=0.001;
    parameter.Sen_T{ll}(i,j)=parameter.Sen_T{ll}(i,j)+e;
    result=Forward(current_batch,parameter,1);
    [cost1,grad]=softmax(result,current_batch,parameter);
    parameter.Sen_T{ll}(i,j)=parameter.Sen_T{ll}(i,j)-2*e;
    result=Forward(current_batch,parameter,1);
    [cost2,grad]=softmax(result,current_batch,parameter);
    parameter.Sen_T{ll}(i,j)=parameter.Sen_T{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end



function[norm]=computeGradNorm(grad,parameter)
    % compute gradient norm
    norm=0;
    for ii=1:parameter.layer_num
        norm=norm+double(sum(grad.Word_S{ii}(:).^2));
        norm=norm+double(sum(grad.Word_T{ii}(:).^2));
        norm=norm+double(sum(grad.Sen_S{ii}(:).^2));
        norm=norm+double(sum(grad.Sen_T{ii}(:).^2));
    end
    norm=sqrt(norm);
end
