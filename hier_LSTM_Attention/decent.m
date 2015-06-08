function[thread]=decent(node,parameter)
    thread.U=node.minus_Prob*node.vector';
    L=parameter.U'*node.minus_Prob;
    thread.left_W=reshape(decent_left_W(node,parameter)*L,parameter.dimension,parameter.dimension)';
    thread.right_W=reshape(decent_right_W(node,parameter)*L,parameter.dimension,parameter.dimension)';
    thread.v=reshape(decent_v(node,parameter,node.Word)*L,parameter.dimension,length(node.Word))';

end

function[current]=decent_v(node,parameter,V)
    position=find(V==node.word_index);
    if node.is_begin==1
        current=zeros(length(V)*parameter.dimension,parameter.dimension);
        current(parameter.dimension*(position-1)+1:parameter.dimension*position,:)=parameter.right_W'.*repmat(node.deri_vector',parameter.dimension,1);
    else
        pre_node=node.pre_node;
        leftmatrix=decent_v(pre_node,parameter,V);
        current=leftmatrix*parameter.left_W';
        current(parameter.dimension*(position-1)+1:parameter.dimension*position,:)=current(parameter.dimension*(position-1)+1:parameter.dimension*position,:)+parameter.right_W';
        current=current.*repmat(node.deri_vector',length(V)*parameter.dimension,1);
    end
        
end

function[current]=decent_right_W(node,parameter)
    % gradient decent for right_W
    G=parameter.vect(node.word_index,:)';
    A=[];
    B = cell(1,parameter.dimension);
    B(:)={G};
    A=blkdiag(B{:});
    if node.is_begin==1
        current=A.*repmat(node.deri_vector',parameter.dimension*parameter.dimension,1); 
    else
        pre_node=node.pre_node;
        leftmatrix=decent_right_W(pre_node,parameter);
        l=leftmatrix*parameter.left_W'+A;
        current=l.*repmat(node.deri_vector',parameter.dimension*parameter.dimension,1);
     end
end

function[current]=decent_left_W(node,parameter)
    if node.is_begin==1
        current=zeros(parameter.dimension*parameter.dimension,parameter.dimension);
    else
        pre_node=node.pre_node;
        G=pre_node.vector;
        A=[];
        B = cell(1,parameter.dimension);
        B(:)={G};
        A=blkdiag(B{:});
        leftmatrix=decent_left_W(pre_node,parameter);
        l=leftmatrix*parameter.left_W'+A;
        current=l.*repmat(node.deri_vector',parameter.dimension*parameter.dimension,1);
    end
end
