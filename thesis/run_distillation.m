% =========================================================================
% 知识蒸馏主程序 (Knowledge Distillation)
% 将 Python 云端大模型的“软标签”转移至底层 C/MATLAB 手工重构的学生网络中
% =========================================================================
clear; clc;

fprintf('1. 正在读取物理数据与云端软标签...\n');
load('training_data.mat');
load('teacher_soft_labels.mat');

features = data_collect.features;       % [N, 19]
targets_hard = data_collect.targets;    % 真实需要的物理补偿电压

% --- 时序对齐 (极其重要) ---
seq_len = 16;
 
Y_soft = double(soft_labels(:));        
N_soft = length(Y_soft);  % 获取 Python 实际吐出的标签数量 (59584)

% 严格以 Python 产出的数量向后截取硬标签和特征，杜绝 off-by-one 错误
X_aligned = features(seq_len : seq_len + N_soft - 1, :);   
Y_hard = targets_hard(seq_len : seq_len + N_soft - 1);     
Y_hard = Y_hard(:);                     % 保证是列向量

assert(length(Y_hard) == length(Y_soft), '警告：硬标签与软标签时间维度不对齐！');

% --- 核心：Hinton 知识蒸馏方程 ---
% alpha = 0.85 意味着学生网络将 85% 的精力用于学习大模型的平滑规律
% 15% 的精力用于锚定真实的物理世界噪声
alpha = 0.85; 
Y_blended = alpha * Y_soft + (1 - alpha) * Y_hard;

fprintf('2. 正在初始化手搓版轻量级学生模型 (2层感知机)...\n');
input_dim = 19;
hidden_dims = [64, 32];  % 极其精简的结构，适合边缘 DSP 部署
nn = ResidualNN_Simple(input_dim, hidden_dims);

fprintf('3. 开始执行纯手工矩阵反向传播 (蒸馏训练)...\n');
% 纯矩阵运算极快，跑 5000 步
epochs = 5000;  
lr = 0.001;      % 学习率
tic;
nn.train_distillation(X_aligned, Y_blended, epochs, lr);
toc;

fprintf('4. 固化参数并导出部署包...\n');
save('trained_nn.mat', 'nn');

fprintf('\n🎉 蒸馏结束！您的手搓学生网络已掌握作动器的迟滞暗知识！\n');
fprintf('👉 终极行动：请前往 main_simulation.m，将 MODE 设为 2 并运行！\n');