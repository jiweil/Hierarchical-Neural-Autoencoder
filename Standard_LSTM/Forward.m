function[lstms,all_h_t,all_c_t]=Forward(batch,parameter,isTraining)%Forward
    N=size(batch.Word,1);
    zeroState=zeroMatrix([parameter.hidden,N]);
    if isTraining==1
        T=batch.MaxLen;
    else
        T=batch.MaxLenSource;
    end
    all_h_t=cell(parameter.layer_num,T);
    all_c_t=cell(parameter.layer_num,T);
    lstms = cell(parameter.layer_num,T);

    for ll=1:parameter.layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
            all_c_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
        end
    end
    for t=1:T
        for ll=1:parameter.layer_num
            if t<batch.MaxLenSource+1;
                W=parameter.W_S{ll};
            else
                W=parameter.W_T{ll};
            end
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 = all_c_t{ll, t-1};
                h_t_1 = all_h_t{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.Word(:,t));
            else
                x_t=all_h_t{ll-1,t};
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            [lstms{ll, t},all_h_t{ll, t},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);%LSTM unit calculation
        end
    end
end


