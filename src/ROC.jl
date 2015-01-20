function AUC_ROC(Ytrue, scores)
     perm = sortperm(scores)
     
     roc_y = Ytrue[perm]
     stack_x = cumsum(roc_y)/sum(roc_y)
     stack_y = cumsum(! roc_y)/sum(! roc_y)    
     N = length(roc_y)
     
     auc = sum( (stack_x[2:N] - stack_x[1:N-1]) .* stack_y[2:N] )
     return auc
end
