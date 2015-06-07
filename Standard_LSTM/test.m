function[]=test()
clear;
%matlabpool open 16
addpath('../misc');
n= gpuDeviceCount;
parameter.isGPU = 0;

if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(2);
else
    print('no gpu ! ! ! ! !');
end

parameter.dimension=1000;
parameter.alpha=0.1;
parameter.layer_num=4;
parameter.hidden=1000;
parameter.lstm_out_tanh=0;
parameter.Initial=0.08;
parameter.dropout=0.2;
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=0;
parameter.update_embedding=1;
parameter.batch_size=32;
parameter.Source_Target_Same_Language=1;
parameter.maxGradNorm=1;
parameter.clip=1;

parameter.lr=5;
parameter.read=1;

if parameter.Source_Target_Same_Language==1
    parameter.Vocab=25001;
    parameter.stop=parameter.Vocab;
else
    parameter.SourceVocab=20;
    parameter.TargetVocab=20;
    parameter.stop=parameter.TargetVocab;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
end

if parameter.CheckGrad==1&parameter.dropout~=0
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end
%alpha: learning rate for minibatch

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;


train_source_file='../data/train_permute.txt';
train_target_file='../data/train_permute.txt';
test_source_file='../data/test.txt';
test_target_file='../data/test.txt';

if parameter.read==1
    disp('read');
    parameter=ReadParameter(parameter);
else [parameter]=Initial(parameter);
end
Test=ReadTestData(test_source_file,parameter);
TestBatches=GetTestBatch(Test,32,parameter);
disp('decode begin')%greedy decoding
decode_greedy(parameter,TestBatches,'a.txt');
end

function[Batches]=GetTestBatch(Source,batch_size,parameter)
    N_batch=ceil(length(Source)/batch_size);
    Batches={};
    for i=1:N_batch
        Begin=batch_size*(i-1)+1;
        End=batch_size*i;
        if End>length(Source)
            End=length(Source);
        end
        current_batch=Batch();
        for j=Begin:End
            source_length=length(Source{j});
            current_batch.SourceLength=[current_batch.SourceLength,source_length];
            if source_length>current_batch.MaxLenSource
                current_batch.MaxLenSource=source_length;
            end
        end
        current_batch.Word=ones(End-Begin+1,current_batch.MaxLenSource);
        Mask=ones(End-Begin+1,current_batch.MaxLenSource);
        for j=Begin:End
            source_length=length(Source{j});
            current_batch.Word(j-Begin+1,current_batch.MaxLenSource-source_length+1:current_batch.MaxLenSource)=Source{j};
            Mask(j-Begin+1,1:current_batch.MaxLenSource-source_length)=0;
        end
        for j=1:current_batch.MaxLenSource
            current_batch.Delete{j}=find(Mask(:,j)==0);
            current_batch.Left{j}=find(Mask(:,j)==1);
        end
        current_batch.Mask=Mask;
        Batches{i}=current_batch;
    end
end


function[Source]=ReadTestData(source_file,parameter)
    fd_s=fopen(source_file);
    Source={};
    i=0;
    tline_s = fgets(fd_s);
    while ischar(tline_s)
        i=i+1;
        text_s=deblank(tline_s);
        if parameter.Source_Target_Same_Language~=1
            Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;
        else
            Source{i}=wrev(str2num(text_s));
        end
        tline_s = fgets(fd_s);
    end
end

function[parameter]=ReadParameter(parameter)
    for ll=1:parameter.layer_num
        W_file=strcat('save_parameter/_W_S',num2str(ll));
        parameter.W_S{ll}=gpuArray(load(W_file));
        W_file=strcat('save_parameter/_W_T',num2str(ll));
        parameter.W_T{ll}=gpuArray(load(W_file));
    end
    parameter.vect=gpuArray(load('save_parameter/_v'));
    parameter.soft_W=gpuArray(load('save_parameter/_soft_W'));
end

