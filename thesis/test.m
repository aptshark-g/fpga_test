load('training_data.mat');
y = data_collect.targets(:);
fprintf('目标均值 = %.4f, 标准差 = %.4f\n', mean(y), std(y));
fprintf('目标方差 = %.4f\n', var(y));
fprintf('基线 MSE (预测均值) = %.4f\n', mean((y - mean(y)).^2));