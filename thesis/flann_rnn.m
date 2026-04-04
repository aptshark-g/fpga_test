% =====================================================================
% 三种控制器对比：FxLMS、FLANN+RNN、自适应混合（基于 THD 加权融合）
% 完整脚本，无 persistent，无 classdef，可直接运行
% =====================================================================

clear; clc; close all;

%% 1. 系统参数
fs = 10000;                 % 采样率 [Hz]
duration = 20;              % 仿真时长 [s]
n_samples = fs * duration;
t = (0:n_samples-1)' / fs;

% 参考信号（100Hz 正弦波 + 轻微噪声）
f_ref = 100;
x_ref = sin(2*pi*f_ref*t) + 0.01*randn(n_samples,1);

% 非线性主路径参数
alpha_plant = 1000;         % 一阶极点 1000 rad/s
dt = 1/fs;

% 次级路径（线性 FIR 滤波器，列向量）
s_hat = [0, 0, 0, 0, -0.8]';
S_real = s_hat;
S_est = S_real;

% 控制器参数
L = 128;                    % FxLMS 阶数
mu_fxlms = 1e-7;            % FxLMS 步长

% FLANN+RNN 参数
flann_order = 10;           % 三角函数扩展阶数
rnn_hidden = 9;             % RNN 隐藏层神经元数
learning_rate = 0.02;       % 在线学习率
flann_input_dim = 1 + 2*flann_order + 1;   % 输入维度 = 12

% 自适应混合参数
thd_window = 256;           % THD 计算窗口长度
thd_update_interval = 20;   % 每 20 步更新一次混合系数
% THD 阈值（根据经验设定）
thd_linear = 0.03;          % THD < 3% 视为线性
thd_weak = 0.08;            % 3%~8% 弱非线性
thd_strong = 0.15;          % >15% 强非线性

%% 2. 辅助函数定义（本地函数）

% FLANN 扩展函数（返回列向量）
flann_expand = @(x, order) [x; reshape([sin((1:order)*pi*x); cos((1:order)*pi*x)], [], 1)];

% 非线性主路径响应（更新状态并返回输出）
% 输入: u, state, alpha, dt
% 输出: new_state, disturbance
nonlinear_plant = @(u, state, alpha, dt) deal( ...
    state + (-alpha * state + alpha * u) * dt, ...
    3 * tanh((state + (-alpha * state + alpha * u) * dt) / 3) );

% THD 计算函数（放在脚本末尾）
function thd = compute_thd(signal, fs, window_len)
    if length(signal) < window_len
        thd = 0;
        return;
    end
    seg = signal(end-window_len+1:end);
    seg = seg .* hamming(window_len);
    Y = fft(seg);
    Y = abs(Y(1:floor(window_len/2)+1));
    [~, idx_max] = max(Y(2:end));
    fundamental = Y(idx_max+1);
    harm_idx = (2:10) * (idx_max+1);
    harm_idx = harm_idx(harm_idx <= length(Y));
    harmonic_power = sqrt(sum(Y(harm_idx).^2));
    thd = harmonic_power / fundamental;
end

%% 3. 预分配结果数组
y_fxlms = zeros(n_samples,1);   err_fxlms = zeros(n_samples,1);
y_nn    = zeros(n_samples,1);   err_nn    = zeros(n_samples,1);
y_hybrid= zeros(n_samples,1);   err_hybrid= zeros(n_samples,1);

%% 4. 运行 FxLMS 控制（基准）
fprintf('运行 FxLMS...\n');

% 初始化状态
w_fxlms = zeros(L,1);
x_buffer = zeros(L,1);
xf_buffer = zeros(L,1);
sec_buf = zeros(length(S_real),1);   % 次级路径滤波器缓冲区
plant_state = 0;

for k = 1:n_samples
    xk = x_ref(k);
    % 更新参考信号缓冲区
    x_buffer = [xk; x_buffer(1:end-1)];
    % 滤波参考信号
    xf = S_est' * x_buffer(1:length(S_est));
    xf_buffer = [xf; xf_buffer(1:end-1)];
    % 控制输出
    y_fxlms(k) = w_fxlms' * x_buffer(1:L);
    % 经过次级路径
    sec_buf = [y_fxlms(k); sec_buf(1:end-1)];
    sec_out = sec_buf' * S_real;
    % 经过主路径
    [plant_state, disturbance] = nonlinear_plant(sec_out, plant_state, alpha_plant, dt);
    err_fxlms(k) = xk - disturbance;
    % 更新 FxLMS 系数
    w_fxlms = w_fxlms - mu_fxlms * err_fxlms(k) * xf_buffer;
end
fprintf('FxLMS 完成。\n');

%% 5. 运行 FLANN+RNN 控制
fprintf('运行 FLANN+RNN...\n');

% 固定随机种子以保证可重复性
rng(1);
% 初始化权重
W_ih = randn(rnn_hidden, flann_input_dim + 1) * 0.1;
W_hh = randn(rnn_hidden, rnn_hidden) * 0.1;
W_ho = randn(1, rnn_hidden + 1) * 0.1;

h_state = zeros(rnn_hidden,1);
sec_buf_nn = zeros(length(S_real),1);
plant_state_nn = 0;
err_prev = 0;

for k = 1:n_samples
    xk = x_ref(k);
    % 输入特征：参考信号 + 上一时刻误差
    phi_ref = flann_expand(xk, flann_order);
    phi = [phi_ref; err_prev];
    % RNN 前向传播
    rnn_input = [phi; 1];
    h_state = tanh(W_ih * rnn_input + W_hh * h_state);
    out_input = [h_state; 1];
    y_nn(k) = W_ho * out_input;
    y_nn(k) = max(min(y_nn(k), 5), -5);
    % 经过次级路径
    sec_buf_nn = [y_nn(k); sec_buf_nn(1:end-1)];
    sec_out = sec_buf_nn' * S_real;
    % 经过主路径
    [plant_state_nn, disturbance] = nonlinear_plant(sec_out, plant_state_nn, alpha_plant, dt);
    err_nn(k) = xk - disturbance;
    err_prev = err_nn(k);
    % 在线更新输出层权重
    delta_out = err_nn(k) * 0.01;
    W_ho = W_ho - learning_rate * delta_out * out_input';
end
fprintf('FLANN+RNN 完成。\n');

%% 6. 运行自适应混合控制
fprintf('运行自适应混合控制...\n');

% 重新初始化 FxLMS 部分
w_hyb_f = zeros(L,1);
x_buf_hyb = zeros(L,1);
xf_buf_hyb = zeros(L,1);
sec_buf_f = zeros(length(S_real),1);
plant_f = 0;

% 重新初始化 NN 部分（与步骤5相同的初始权重）
rng(1);
W_ih_h = randn(rnn_hidden, flann_input_dim + 1) * 0.1;
W_hh_h = randn(rnn_hidden, rnn_hidden) * 0.1;
W_ho_h = randn(1, rnn_hidden + 1) * 0.1;
h_state_h = zeros(rnn_hidden,1);
sec_buf_n = zeros(length(S_real),1);
plant_n = 0;
err_hyb_prev = 0;

% 混合系数 alpha
alpha = 0.1;
% 用于混合控制的最终输出缓冲区（实际施加到系统）
sec_buf_total = zeros(length(S_real),1);
plant_total = 0;

% 用于存储 THD 历史（可选）
thd_history = [];

for k = 1:n_samples
    xk = x_ref(k);
    
    % ---------- FxLMS 部分 ----------
    x_buf_hyb = [xk; x_buf_hyb(1:end-1)];
    xf = S_est' * x_buf_hyb(1:length(S_est));
    xf_buf_hyb = [xf; xf_buf_hyb(1:end-1)];
    y_f = w_hyb_f' * x_buf_hyb(1:L);
    sec_buf_f = [y_f; sec_buf_f(1:end-1)];
    sec_out_f = sec_buf_f' * S_real;
    [plant_f, ~] = nonlinear_plant(sec_out_f, plant_f, alpha_plant, dt);
    
    % ---------- NN 部分 ----------
    phi_ref = flann_expand(xk, flann_order);
    phi = [phi_ref; err_hyb_prev];
    rnn_input = [phi; 1];
    h_state_h = tanh(W_ih_h * rnn_input + W_hh_h * h_state_h);
    out_input = [h_state_h; 1];
    y_n = W_ho_h * out_input;
    y_n = max(min(y_n, 5), -5);
    sec_buf_n = [y_n; sec_buf_n(1:end-1)];
    sec_out_n = sec_buf_n' * S_real;
    [plant_n, ~] = nonlinear_plant(sec_out_n, plant_n, alpha_plant, dt);
    
    % ---------- 混合输出 ----------
    y_hybrid(k) = alpha * y_f + (1-alpha) * y_n;
    y_hybrid(k) = max(min(y_hybrid(k), 5), -5);
    
    % 将混合输出施加到真实系统（使用单独的缓冲区）
    sec_buf_total = [y_hybrid(k); sec_buf_total(1:end-1)];
    sec_out_total = sec_buf_total' * S_real;
    [plant_total, disturbance] = nonlinear_plant(sec_out_total, plant_total, alpha_plant, dt);
    err_hybrid(k) = xk - disturbance;
    err_hyb_prev = err_hybrid(k);
    
    % ---------- 更新 FxLMS 和 NN 权重（使用混合误差）----------
    w_hyb_f = w_hyb_f - mu_fxlms * err_hybrid(k) * xf_buf_hyb;
    delta_out = err_hybrid(k) * 0.01;
    W_ho_h = W_ho_h - learning_rate * delta_out * out_input';
    
    % ---------- 更新混合系数 alpha（基于当前误差的 THD）----------
    if mod(k, thd_update_interval) == 0 && k > thd_window
        thd = compute_thd(err_hybrid(max(1,k-thd_window+1):k), fs, thd_window);
        thd_history(end+1) = thd;
        if thd < thd_linear
            alpha = 1.0;
        elseif thd < thd_weak
            alpha = 1 - (thd - thd_linear)/(thd_weak - thd_linear) * 0.5;
        elseif thd < thd_strong
            alpha = 0.5 - (thd - thd_weak)/(thd_strong - thd_weak) * 0.3;
        else
            alpha = 0.2;
        end
        alpha = max(0, min(1, alpha));
    end
end
fprintf('自适应混合控制完成。\n');

%% 7. 性能评估与对比
rms_f = rms(err_fxlms);   rms_n = rms(err_nn);   rms_h = rms(err_hybrid);
red_f = 10*log10(rms(x_ref)^2 / rms_f^2);
red_n = 10*log10(rms(x_ref)^2 / rms_n^2);
red_h = 10*log10(rms(x_ref)^2 / rms_h^2);

thd_f = compute_thd(err_fxlms, fs, 256);
thd_n = compute_thd(err_nn, fs, 256);
thd_h = compute_thd(err_hybrid, fs, 256);

fprintf('\n========== 最终性能对比 ==========\n');
fprintf('FxLMS         : 抑制比 = %.2f dB, 残余 THD = %.2f%%\n', red_f, thd_f*100);
fprintf('FLANN+RNN     : 抑制比 = %.2f dB, 残余 THD = %.2f%%\n', red_n, thd_n*100);
fprintf('自适应混合    : 抑制比 = %.2f dB, 残余 THD = %.2f%%\n', red_h, thd_h*100);
fprintf('混合相比 FxLMS 提升: %.2f dB\n', red_h - red_f);
fprintf('混合相比 NN 提升    : %.2f dB\n', red_h - red_n);

%% 8. 绘图对比
figure('Name', '三种控制器性能对比', 'Position', [100, 100, 1400, 900]);

subplot(2,2,1);
plot(t, err_fxlms, 'b', t, err_nn, 'r', t, err_hybrid, 'g');
xlabel('时间 (s)'); ylabel('误差 (m/s²)');
legend('FxLMS', 'FLANN+RNN', '自适应混合');
title('残余误差时域对比');
grid on; xlim([0, 2]);

subplot(2,2,2);
semilogy(t, abs(err_fxlms), 'b', t, abs(err_nn), 'r', t, abs(err_hybrid), 'g');
xlabel('时间 (s)'); ylabel('|误差|');
legend('FxLMS', 'FLANN+RNN', '自适应混合');
title('绝对误差（对数坐标）');
grid on; xlim([0, 2]);

subplot(2,2,3);
[Pf, f] = pwelch(err_fxlms, hamming(512), 256, 1024, fs);
[Pn, ~] = pwelch(err_nn, hamming(512), 256, 1024, fs);
[Ph, ~] = pwelch(err_hybrid, hamming(512), 256, 1024, fs);
semilogy(f, Pf, 'b', f, Pn, 'r', f, Ph, 'g');
xlabel('频率 (Hz)'); ylabel('功率谱密度');
legend('FxLMS', 'FLANN+RNN', '自适应混合');
title('残余误差频谱');
xlim([0, 1000]); grid on;

subplot(2,2,4);
bar([red_f, red_n, red_h]);
set(gca, 'XTickLabel', {'FxLMS', 'FLANN+RNN', '自适应混合'});
ylabel('抑制比 (dB)');
title('抑制比对比');
grid on;

save('three_controllers_results.mat', 't', 'err_fxlms', 'err_nn', 'err_hybrid', ...
     'red_f', 'red_n', 'red_h', 'thd_f', 'thd_n', 'thd_h');
fprintf('\n结果已保存至 three_controllers_results.mat\n');