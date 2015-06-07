function[lstm, h_t, c_t]=lstmUnit(W,parameter,x_t,h_t_1, c_t_1, ll, t,isTraining)%lstm unit calculation
    if parameter.dropout~=0&&isTraining==1
        if parameter.CheckGrad==1   
            if ll==1
                drop_left=repmat(parameter.drop_left_1,1,size(x_t,2));
            else
                drop_left=repmat(parameter.drop_left,1,size(x_t,2));
            end
        else
            drop_left=randSimpleMatrix(size(x_t))<1-parameter.dropout;
        end
        x_t=x_t.*drop_left;
    end
    input=[x_t; h_t_1];
    ifoa_linear = W*input;
    ifo_gate=parameter.nonlinear_gate_f(ifoa_linear(1:3*parameter.hidden,:));
    i_gate = ifo_gate(1:parameter.hidden, :);
    f_gate = ifo_gate(parameter.hidden+1:2*parameter.hidden,:);
    o_gate =ifo_gate(parameter.hidden*2+1:3*parameter.hidden,:);
    a_signal = parameter.nonlinear_f(ifoa_linear(3*parameter.hidden+1:4*parameter.hidden,:));
    c_t=f_gate.*c_t_1 + i_gate.*a_signal;
    if parameter.lstm_out_tanh
        f_c_t = parameter.nonlinear_f(c_t);
        h_t = o_gate.*f_c_t;
    else
        h_t = o_gate.*c_t;
    end
    if parameter.clip==1%if values are too large, clip the values.
        c_t = arrayfun(@clipForward, c_t);
        h_t = arrayfun(@clipForward, h_t);
    end

    lstm.input = input;
    lstm.i_gate = i_gate;
    lstm.f_gate = f_gate;
    lstm.o_gate = o_gate;
    lstm.a_signal = a_signal;
    if parameter.lstm_out_tanh
        lstm.f_c_t = f_c_t;
    else
        lstm.c_t = c_t;
    end
    if isTraining==1&&parameter.dropout~=0
        lstm.drop_left=drop_left;
    end
end

function [clippedValue] = clipForward(x)
    if x>50 clippedValue = single(50);
    elseif x<-50 clippedValue = single(-50);
    else clippedValue =single(x);
    end
end

