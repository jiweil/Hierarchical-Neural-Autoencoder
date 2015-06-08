function[]=Tune()
clear;
%matlabpool open 16
addpath('./misc');
parameter.dimension=25;
parameter.hidden=25;
parameter.lstm_out_tanh=0;
parameter.Initial=0.1;
parameter.class=5;
parameter.dropout=0;
parameter.check_Grad=1;
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=0;
parameter.update_embedding=1;


if parameter.CheckGrad==1&parameter.dropout~=0
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end
%alpha: learning rate for minibatch

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;

n= gpuDeviceCount;
parameter.isGPU = 0;

if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(2);
else
    print('no gpu ! ! ! ! !');
end

parameter.C=1.0/power(10,6);

train_file='../sequence_train1.txt';
dev_file='../sequence_dev.txt';
test_file='../sequence_test1.txt';
test_file='../sequence_test_root1.txt';
test_root_file='../sequence_test_root.txt';


iter=0;
[TrainTag,TrainNode]=ReadData(train_file);
[TestTag,TestNode]=ReadData(test_file);
disp('read data done');

ALPHA=[0.05,0.1,0.25,0.5];
SIZE=[100,500,1000,2500,5000];
LAYER=[1,2,3,4];

parameter.alpha=ALPHA(2);
parameter.mini_batch_size=SIZE(2);
TrainBatches=GetBatch(TrainTag,TrainNode,parameter.mini_batch_size,parameter);
TestBatches=GetBatch(TestTag,TestNode,parameter.mini_batch_size,parameter);


result=[];
for layer=1:length(LAYER)
    for i=1:length(ALPHA)
        for j=1:length(SIZE)
            [layer,i,j]
            parameter.alpha=ALPHA(i);
            parameter.mini_batch_size=SIZE(j);
            parameter.layer_num=LAYER(layer);

            [parameter,ada]=Initial(parameter); %intial parameter
            Acc=Model(TrainBatches,TestBatches,parameter,ada);
            result=[result;[Acc,parameter.layer_num,parameter.alpha,parameter.mini_batch_size]];  
        end
    end
end
dlmwrite('result',result,'delimiter',' ','precision',5)

end


function[Acc]=Model(TrainBatches,TestBatches,parameter,ada)
    max_iter=10;
    accuracy_matrix=[];
    for iter=1:max_iter
        G=randperm(length(TrainBatches));
        %G=1:length(TrainBatches);
        for j=1:length(TrainBatches)
            batch=TrainBatches{G(j)};
            %batch.Label
            [grad,cost,prediction]=Forward(batch,parameter,1);
            [parameter,ada]=update_parameter(parameter,ada,grad);
        end
        accuracy_matrix=[accuracy_matrix,Testing(TestBatches,parameter)];
        if iter==15
            break;
        end
    end
    Acc=max(accuracy_matrix);
end

function[accuracy]=Testing(TestBatches,parameter)
    correct=0;
    total=0;
    Cost=0;
    for j=1:length(TestBatches)
        batch=TestBatches{j};
        %batch.Label
        [grad,cost,prediciton]=Forward(batch,parameter,0);
        correct=correct+sum(prediciton==batch.Label);
        total=total+length(prediciton);
        Cost=Cost+cost;
    end
    accuracy=correct/total;
end

function[parameter,ada]=update_parameter(parameter,ada,grad)
    for i=1:parameter.layer_num
        ada.W{i}=ada.W{i}+grad.W{i}.^2;
        L=find(ada.W{i}~=0);
        parameter.W{i}(L)=parameter.W{i}(L)-parameter.alpha*grad.W{i}(L)./sqrt(ada.W{i}(L));
    end
    ada.U=ada.U+grad.U.^2;
    L=find(ada.U~=0);
    parameter.U(L)=parameter.U(L)-parameter.alpha*grad.U(L)./sqrt(ada.U(L));

    ada.vect(:,grad.indices)=ada.vect(:,grad.indices)+grad.W_emb.^2;
    parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-parameter.alpha*grad.W_emb./sqrt(ada.vect(:,grad.indices));
end

function[parameter,ada]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        if i==1
            parameter.W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
            ada.W{i}=zeroMatrix([4*parameter.hidden,parameter.dimension+parameter.hidden]);
        else
            parameter.W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
            ada.W{i}=zeroMatrix([4*parameter.hidden,2*parameter.hidden]);
        end
    end
    parameter.U=randomMatrix(parameter.Initial,[parameter.class,parameter.hidden]);
    ada.U=zeroMatrix([parameter.class,parameter.hidden]);
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,19539]);
    parameter.vect(:,19538)=zeros([parameter.dimension,1]);
    ada.vect=zeroMatrix([parameter.dimension,19539]);
end

function[parameter]=ReadVect(filename,parameter)
    %read pre-trained vectors from file
    fd=fopen(filename);
    tline = fgets(fd);
    i=0;
    while ischar(tline)
        i=i+1;
        tline=deblank(tline);
        parameter.vect(i,:)=str2num(deblank(tline));
        tline = fgets(fd);
    end
end


function[Batches]=GetBatch(Tag,Word,batch_size,parameter)
    N_batch=ceil(length(Word)/batch_size);
    Batches={};
    for i=1:N_batch
        Begin=batch_size*(i-1)+1;
        End=batch_size*i;
        if End>length(Word)
            End=length(Word);
        end
        current_batch=Batch();
        current_batch.Label=zeros(1,End-Begin+1);
        current_batch.MaxLen=length(Word{End});
        current_batch.Word=ones(End-Begin+1,current_batch.MaxLen);
        Delete=zeros(End-Begin+1,current_batch.MaxLen);
        for j=Begin:End
            leng=length(Word{j});
            current_batch.Word(j-Begin+1,current_batch.MaxLen-leng+1:current_batch.MaxLen)=Word{j};
            Delete(j-Begin+1,1:current_batch.MaxLen-leng)=1;
            current_batch.Label(j-Begin+1)=Tag{j};
        end
        for j=1:current_batch.MaxLen
            current_batch.Delete{j}=find(Delete(:,j)==1);
            current_batch.Left{j}=find(Delete(:,j)==0);
        end
        Batches{i}=current_batch;
    end
end

function[Tag,Word]=ReadData(filename)
    fd=fopen(filename);
    tline = fgets(fd);
    i=0;
    Tag={};Word={};
    while ischar(tline)
        i=i+1;
        text=deblank(tline);
        Tag{i}=str2num(text(1));
        text=text(3:length(text));
        Word{i}=str2num(text);
        tline = fgets(fd);
    end
end

function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [grad,cost1,prediction]=Forward(batch,parameter,0);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [grad,cost2,prediction]=Forward(batch,parameter,0);
    parameter.vect(i,j)=parameter.vect(i,j)+e;

    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_W(value1,ll,i,j,batch,parameter)
    e=0.001;
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    [grad,cost1,prediction]=Forward(batch,parameter,0);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)-2*e;
    [grad,cost2,prediction]=Forward(batch,parameter,0);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_U(value1,i,j,batch,parameter)
    e=0.0001;
    parameter.U(i,j)=parameter.U(i,j)+e;
    [grad,cost1,prediction]=Forward(batch,parameter,0);
    parameter.U(i,j)=parameter.U(i,j)-2*e;
    [grad,cost2,prediction]=Forward(batch,parameter,0);
    parameter.U(i,j)=parameter.U(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1-value2
end
