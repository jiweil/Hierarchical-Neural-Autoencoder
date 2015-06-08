function[]=test()
clear;

addpath('../misc');
n= gpuDeviceCount;
parameter.isGPU = 0;

if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(2);
else
    print('no gpu ! ! ! ! !');
end

parameter.dimension=100;
parameter.alpha=0.1;  %learning rate
parameter.layer_num=4;  %number of layer
parameter.hidden=100;
parameter.lstm_out_tanh=0;
parameter.Initial=0.1;
parameter.dropout=0.2;  %drop-out rate
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;   %whether check gradient or not.
parameter.PreTrainEmb=0;    
%whether using pre-trained embeddings
parameter.update_embedding=1;
%whether update word embeddings
parameter.batch_size=32;
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
        parameter.Sen_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
    end
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,parameter.Vocab]);
    if parameter.Source_Target_Same_Language==1
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.Vocab,parameter.hidden]);
    else
        parameter.soft_W=randomMatrix(parameter.Initial,[parameter.TargetVocab,parameter.hidden]);
    end
end

