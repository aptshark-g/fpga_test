% =====================================================================
% FLANN+RNN 模块专项测试 (终极修正版：带 Filtered-X 权值更新)
% 解决次级路径相位延迟导致的神经网络高频激振与发散问题
% =====================================================================
clear; clc; close all;

%% 1. 系统参数与目标生成
fs = 10000;                 
duration = 10;              
n_samples = fs * duration;
t = (0:n_samples-1)' / fs;

f_ref = 100;
x_ref = sin(2*pi*f_ref*t);

% 模拟解耦后的非线性残差 (包含 200Hz, 300Hz 和白噪声)
d_nonlinear = 0.2 * sin(2*pi*2*f_ref*t + pi/4) + 0.1 * sin(2*pi*3*f_ref*t - pi/3) + 0.01*randn(n_samples,1);
[b_band, a_band] = butter(4, [150, 350]/(fs/2), 'bandpass');
d_target = filtfilt(b_band, a_band, d_nonlinear);

%% 2. 物理路径参数
alpha_plant = 1000;         
dt = 1/fs;
S_real = [0, 0, 0, -0.8]';  

% FLANN+RNN 参数
flann_order = 5;            
rnn_hidden = 8;             
learning_rate = 0.005;        
flann_input_dim = 1 + 2*flann_order + 1; 

% 辅助函数
flann_expand = @(x, order) [x; reshape([sin((1:order)*pi*x); cos((1:order)*pi*x)], [], 1)];
nonlinear_plant = @(u, state, alpha, dt) deal( ...
    state + (-alpha * state + alpha * u) * dt, ...
    3 * tanh((state + (-alpha * state + alpha * u) * dt) / 3) );

%% 3. 【核心修正】辨识完整的次级路径脉冲响应 (含植物相位)
impulse_len = 64;
S_est_full = zeros(impulse_len, 1);
test_u = [1; zeros(impulse_len-1, 1)]; % 单位脉冲
tmp_sec = zeros(length(S_real), 1);
tmp_p = 0;
for i = 1:impulse_len
    tmp_sec = [test_u(i); tmp_sec(1:end-1)];
    [tmp_p, out] = nonlinear_plant(tmp_sec' * S_real, tmp_p, alpha_plant, dt);
    S_est_full(i) = out;
end
% S_est_full 现在包含了系统真实的相位和幅度衰减信息！

%% 4. 初始化网络并运行控制
rng(42);
W_ih = randn(rnn_hidden, flann_input_dim + 1) * 0.1;
W_hh = randn(rnn_hidden, rnn_hidden) * 0.1;
W_ho = randn(1, rnn_hidden + 1) * 0.1;

h_state = zeros(rnn_hidden, 1);
sec_buf_nn = zeros(length(S_real), 1);
plant_state_nn = 0;

err_nn = zeros(n_samples, 1);
y_nn = zeros(n_samples, 1);
err_prev = 0;

% 【核心修正】为 Filtered-X 准备隐藏层输出的缓冲区
out_input_buf = zeros(rnn_hidden + 1, impulse_len);

fprintf('开始运行 Fx-FLANN+RNN 非线性子带补偿...\n');
for k = 1:n_samples
    xk = x_ref(k);       
    dk = d_target(k);    
    
    % --- 前向传播 ---
    phi_ref = flann_expand(xk, flann_order);
    phi = [phi_ref; err_prev];
    rnn_input = [phi; 1];
    
    h_state = tanh(W_ih * rnn_input + W_hh * h_state);
    out_input = [h_state; 1];
    y_nn(k) = W_ho * out_input;
    y_nn(k) = max(min(y_nn(k), 2), -2); 
    
    % --- 物理路径 ---
    sec_buf_nn = [y_nn(k); sec_buf_nn(1:end-1)];
    sec_out = sec_buf_nn' * S_real;
    [plant_state_nn, act_out] = nonlinear_plant(sec_out, plant_state_nn, alpha_plant, dt);
    
    % --- 误差计算 ---
    err_nn(k) = dk - act_out;
    err_prev = err_nn(k);
    
    % --- 【核心修正】Filtered-X 神经网络反向更新 ---
    % 1. 缓存当前的隐藏层特征
    out_input_buf = [out_input, out_input_buf(:, 1:end-1)];
    % 2. 用次级路径过滤特征 (对齐相位)
    out_f = out_input_buf * S_est_full; 
    % 3. 用过滤后的特征更新输出层权重
    W_ho = W_ho + learning_rate * err_nn(k) * out_f';
end
fprintf('仿真完成。\n');

%% 5. 性能量化与展示
rms_orig = rms(d_target);
rms_resid = rms(err_nn(fs*2:end)); % 忽略前2秒暂态，让网络收敛
reduction_dB = 10*log10(rms_orig^2 / rms_resid^2);

fprintf('子带初始 RMS: %.4f\n', rms_orig);
fprintf('补偿后残余 RMS: %.4f\n', rms_resid);
fprintf('非线性频段抑制比: %.2f dB\n', reduction_dB);

%% 6. 绘图分析
figure('Name', 'Fx-FLANN+RNN 性能验证', 'Position', [100, 100, 1000, 800]);

subplot(3,1,1);
plot(t, d_target, 'k', 'LineWidth', 1); hold on;
plot(t, err_nn, 'r', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('幅值 (m/s²)');
title('非线性残差时域补偿效果');
legend('未控制扰动', 'Fx-FLANN+RNN 补偿后');
grid on; xlim([duration-0.1, duration]); 

subplot(3,1,2);
plot(t, y_nn, 'b', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('控制输出');
title('神经网络输出波形');
grid on; xlim([duration-0.1, duration]);

subplot(3,1,3);
[P_orig, f_psd] = pwelch(d_target(fs*2:end), hamming(512), 256, 1024, fs);
[P_resid, ~] = pwelch(err_nn(fs*2:end), hamming(512), 256, 1024, fs);
semilogy(f_psd, P_orig, 'k', 'LineWidth', 1.5); hold on;
semilogy(f_psd, P_resid, 'r', 'LineWidth', 1.5);
xlabel('频率 (Hz)'); ylabel('功率谱密度');
title('频段内能量衰减 (相位对齐后，彻底消除激振毛刺)');
legend('未控制', '补偿后');
grid on; xlim([0, 500]);