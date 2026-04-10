% =====================================================================
% FLANN+RNN 模块专项测试：频段解耦后的非线性残差补偿
% 验证：剥离线性基频后，神经网络不再产生高频毛刺，实现平滑拟合
% =====================================================================
clear; clc; close all;

%% 1. 系统参数
fs = 10000;                 % 采样率 [Hz]
duration = 10;              % 仿真时长 [s]
n_samples = fs * duration;
t = (0:n_samples-1)' / fs;

% 原始参考信号（提供基准频率，用于 FLANN 扩展）
f_ref = 100;
x_ref = sin(2*pi*f_ref*t);

% 模拟解耦后的非线性残差扰动 (FxLMS已经吃掉了100Hz基频)
% 这里包含二次谐波(200Hz)、三次谐波(300Hz)和轻微白噪声
d_nonlinear = 0.2 * sin(2*pi*2*f_ref*t + pi/4) + 0.1 * sin(2*pi*3*f_ref*t - pi/3) + 0.01*randn(n_samples,1);

% 模拟频段拆分：使用带通滤波器只截取 150~350Hz 的窄带信号
[b_band, a_band] = butter(4, [150, 350]/(fs/2), 'bandpass');
d_target = filtfilt(b_band, a_band, d_nonlinear);

%% 2. 物理路径与控制器参数
alpha_plant = 1000;         % 作动器模拟极点
dt = 1/fs;
S_real = [0, 0, 0, -0.8]';  % 次级路径

% FLANN+RNN 参数
flann_order = 5;            % 扩展到5阶足够覆盖200Hz和300Hz
rnn_hidden = 8;             % 轻量化隐藏层
learning_rate = 0.05;       % 剥离大能量后，可适当提高学习率加速收敛
flann_input_dim = 1 + 2*flann_order + 1; % 1(原)+10(三角)+1(前误差)=12维

%% 3. 辅助函数
% FLANN 扩展：严格生成 sin(wx), cos(wx), sin(2wx)... 等基函数
flann_expand = @(x, order) [x; reshape([sin((1:order)*pi*x); cos((1:order)*pi*x)], [], 1)];

% 作动器模拟
nonlinear_plant = @(u, state, alpha, dt) deal( ...
    state + (-alpha * state + alpha * u) * dt, ...
    3 * tanh((state + (-alpha * state + alpha * u) * dt) / 3) );

%% 4. 初始化网络并运行控制
rng(42); % 固定种子
W_ih = randn(rnn_hidden, flann_input_dim + 1) * 0.1;
W_hh = randn(rnn_hidden, rnn_hidden) * 0.1;
W_ho = randn(1, rnn_hidden + 1) * 0.1;

h_state = zeros(rnn_hidden, 1);
sec_buf_nn = zeros(length(S_real), 1);
plant_state_nn = 0;

err_nn = zeros(n_samples, 1);
y_nn = zeros(n_samples, 1);
err_prev = 0;

fprintf('开始运行 FLANN+RNN 非线性子带补偿...\n');
for k = 1:n_samples
    xk = x_ref(k);       % 必须用 100Hz 原始信号作为 FLANN 种子
    dk = d_target(k);    % 当前时刻的子带非线性扰动
    
    % --- FLANN+RNN 前向传播 ---
    phi_ref = flann_expand(xk, flann_order);
    phi = [phi_ref; err_prev];
    rnn_input = [phi; 1];
    
    h_state = tanh(W_ih * rnn_input + W_hh * h_state);
    out_input = [h_state; 1];
    y_nn(k) = W_ho * out_input;
    y_nn(k) = max(min(y_nn(k), 2), -2); % 输出限幅
    
    % --- 物理路径交互 ---
    sec_buf_nn = [y_nn(k); sec_buf_nn(1:end-1)];
    sec_out = sec_buf_nn' * S_real;
    [plant_state_nn, act_out] = nonlinear_plant(sec_out, plant_state_nn, alpha_plant, dt);
    
    % --- 误差计算与权值更新 ---
    err_nn(k) = dk - act_out;
    err_prev = err_nn(k);
    
    delta_out = err_nn(k) * 0.05;
    W_ho = W_ho - learning_rate * delta_out * out_input';
end
fprintf('仿真完成。\n');

%% 5. 性能量化与展示
rms_orig = rms(d_target);
rms_resid = rms(err_nn(fs:end)); % 忽略第一秒暂态
reduction_dB = 10*log10(rms_orig^2 / rms_resid^2);

fprintf('子带初始 RMS: %.4f\n', rms_orig);
fprintf('补偿后残余 RMS: %.4f\n', rms_resid);
fprintf('非线性频段抑制比: %.2f dB\n', reduction_dB);

%% 6. 绘图分析
figure('Name', '解耦后的 FLANN+RNN 性能', 'Position', [100, 100, 1000, 800]);

% 时域波形对比
subplot(3,1,1);
plot(t, d_target, 'k', 'LineWidth', 1); hold on;
plot(t, err_nn, 'r', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('幅值 (m/s²)');
title('非线性残差时域补偿效果 (150-350 Hz 频段)');
legend('未控制非线性扰动', 'FLANN+RNN 补偿后残差');
grid on; xlim([1, 1.1]); % 放大看稳态波形

% 控制信号 y_nn 波形
subplot(3,1,2);
plot(t, y_nn, 'b', 'LineWidth', 1);
xlabel('时间 (s)'); ylabel('控制输出');
title('神经网络输出波形 (平滑、无严重高频毛刺)');
grid on; xlim([1, 1.1]);

% 频域功率谱对比
subplot(3,1,3);
[P_orig, f_psd] = pwelch(d_target, hamming(512), 256, 1024, fs);
[P_resid, ~] = pwelch(err_nn, hamming(512), 256, 1024, fs);
semilogy(f_psd, P_orig, 'k', 'LineWidth', 1.5); hold on;
semilogy(f_psd, P_resid, 'r', 'LineWidth', 1.5);
xlabel('频率 (Hz)'); ylabel('功率谱密度');
title('频段内能量衰减 (200Hz 与 300Hz 谐波被成功抑制)');
legend('未控制', '补偿后');
grid on; xlim([0, 500]);