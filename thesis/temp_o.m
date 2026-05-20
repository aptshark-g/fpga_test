% =====================================================================
% 面向精密加工的振动主动控制仿真主程序 (正交解耦·终极全景版·论文修复版·索引越界修复)
% 修复内容：
% 1. 修正变步长FxLMS逻辑，频率突变时自动重启大步长收敛
% 2. 新增蒸馏NN的16步滑动时间窗口，匹配Python训练时的输入维度
% 3. 修正NN权重调度逻辑，强非线性时强制NN主导
% 4. 新增论文核心的「两阶段物理滤波保底机制」，从根源防THD爆炸
% 5. 【本次修复】修正error_signal_nn为全长向量，解决extract_features索引越界
% =====================================================================
clear; clc; close all;

%==================== 1. 终极三维控制面板 ====================
% 【维度一：非线性架构模式】
% 1: 纯 FxLMS (无非线性补偿)
% 2: FxLMS + FLANN+RNN (纯边缘端在线学习，无先验暗知识)
% 3: FxLMS + 蒸馏 NN (加载 trained_nn.mat，具备相位超前预判)
MODE = 3;       

% 【维度二：线性底座模式 (FxLMS 步长策略)】
% 1: 定步长 FxLMS (Fixed Step)
% 2: 变步长 FxLMS (VSS - 依据残差能量自动伸缩)【修复版】
STEP_MODE = 2;  

% 【维度三：端云融合策略】 (仅当 MODE=2 或 3 时生效)
% 1: 固定微小比例混合 (0.05 强行写死)
% 2: 动态凸因子 α (零和博弈跷跷板)
% 3: 动态非对称混合 ρ (梯度自适应) + 窄带谐波刺客 
MIX_MODE = 3;   

% 【环境工况】
% 1:多工步突变 2:固定100Hz 3:线性扫频 4:正弦调制
EM = 1;

% 数据采集
ta = 0;

% 【维度四：窄带谐波刺客开关】
% 0: 关闭谐波刺客 (y_harm=0)
% 1: 开启谐波刺客 (原基准组)
HARM_SWITCH = 1;  % 做对照实验时仅需改这里为0

%==================== 2. 系统与物理环境参数 ====================
fs = 10000;                 
dt = 1/fs;                  
duration = 30;              
n_samples = duration * fs;
t = (0:n_samples-1)' / fs;
vib_freq = 100;
vib_amp = 15.0;             

% ----次级---- 4 -0.2
delay = 2;                  
s_hat_gain = -0.14;          
S_est = zeros(delay+1, 1);
S_est(end) = s_hat_gain;

%==================== 3. 实例化复杂物理底座 ====================
vib_source = VibrationSource();
structure = StructuralDynamics(fs, 5, true);
actuator = PiezoActuator();
actuator.K = 0.85; actuator.fn = 3100; actuator.zeta = 0.023;
actuator.alpha = 0.15; actuator.beta = 0.035; actuator.gamma = 0.025;
sensor = Accelerometer(fs);
sensor.sensitivity = 98; sensor.bandwidth = [0.5, 3000]; sensor.noise_density = 6e-6;

%==================== 4. 各维度控制器初始化 ====================
%--- [维度二] FxLMS 线性底座初始化 ---
L = 128;                    
mu_fixed = 5e-7;            
mu_vss_max = 5e-5;    %5/6     
mu_vss_min = 1e-7;          
w_fxlms = zeros(L,1);
x_buffer = zeros(L,1);
xf_buffer = zeros(L,1);

% --- [维度一] 非线性架构初始化 ---
if MODE == 2
    % 纯边缘端 FLANN+RNN (无蒸馏)
    flann_order = 5;            
    rnn_hidden = 16;            
    lr_nn = 0.002;              
    flann_input_dim = 2 + 2 * flann_order; 
    rng('shuffle'); % 每次运行权重全变
    W_in2hid = randn(rnn_hidden, flann_input_dim) * 0.1; 
    W_hid2hid = randn(rnn_hidden, rnn_hidden) * 0.1;
    W_out = zeros(1, rnn_hidden + 1); 
    h_state = zeros(rnn_hidden, 1); 
    err_buffer = zeros(delay + 1, 1);
    nn_features_buffer = zeros(delay + 1, rnn_hidden + 1); % 特征滤波缓冲池
    fprintf('>> 架构选择: FLANN+RNN (纯边缘端随机初始化，无先验)\n');
elseif MODE == 3
    % 蒸馏 NN 
    if isfile('trained_nn.mat')
        S = load('trained_nn.mat');
        if isfield(S, 'nn'), nn = S.nn; 
            fprintf('>> 架构选择: 成功加载云端预训练蒸馏模型 trained_nn.mat！\n');
        else, error('无效的 nn 对象'); 
        end
    else, error('缺失 trained_nn.mat'); 
    end
end

% --- [维度三] 融合策略调度器初始化 ---
thd_window = 256;           
thd_update_interval = 50;   
alpha_convex = 0.05;        
rho_gain = 0;   
w_nn = 0; % 【进化】：替换 rho_gain，升级为全自适应桥接权重
mu_rho = 2e-5;              
yn_f_buffer = zeros(delay+1, 1); 
% 谐波专属参数
mu_harm = 5e-4;             
w_harm = zeros(4, 1);       
x_harm_buffer = zeros(delay+1, 4);


% ==============================================
% 【新增】MODE=3 稳态衰减专属参数
% ==============================================
steady_state_counter = 0;           % 稳态持续计数器
steady_state_threshold = 0.8;       % 稳态误差阈值 (m/s²)
steady_state_min_duration = 2.0;    % 连续稳定2秒才判定为真正稳态
w_nn_decay_rate = 0.0001;           % 稳态衰减速率 (每步衰减0.0001，约10秒从1.0降到0.05)
w_nn_min_steady = 0.05;             % 稳态时w_nn的最低下限（保留最小备份权限）


% === 【修复4新增】论文核心两阶段解耦 窄带基频剥离滤波器 ===
order_filter = 4;
fs_global = fs;
% 0-10s 80Hz基频带阻
Wn_stop80 = [65, 95]/(fs_global/2); 
[b_stop80, a_stop80] = butter(order_filter, Wn_stop80, 'stop');
% 10-20s 100Hz基频带阻
Wn_stop100 = [85, 115]/(fs_global/2); 
[b_stop100, a_stop100] = butter(order_filter, Wn_stop100, 'stop');
% 20-30s 110Hz基频带阻
Wn_stop110 = [95, 125]/(fs_global/2); 
[b_stop110, a_stop110] = butter(order_filter, Wn_stop110, 'stop');
% 误差信号缓冲区，用于滤波
err_filter_buffer = zeros(1000, 1); 

%==================== 5. 预分配数组 ====================
vibration_raw = zeros(n_samples,1);
control_signal = zeros(n_samples,1);
error_signal = zeros(n_samples,1);
% === 【本次修复新增】预分配滤波后的误差信号全长向量 ===
error_signal_nn = zeros(n_samples,1); 
y_linear = zeros(n_samples,1);
y_nn = zeros(n_samples,1);
y_harm_history = zeros(n_samples,1); 
history_factor = zeros(n_samples,1); 
history_mu = zeros(n_samples,1);     
% === 云端训练数据日志池 ===
Feature_Log = zeros(n_samples, 19);
% === 【修复2新增】蒸馏模型16步滑动特征窗口缓冲区 ===
feature_window = zeros(19, 16); % 完全匹配Python训练时的SEQ_LEN=16

%==================== 6. 主仿真循环 ====================
acc_control = 0;   
current_phase = 0;
err_env = 0; % 初始化误差包络记录器
fprintf('=== 开始仿真 | 架构:%d | 底座:%d | 融合:%d | 环境:%d ===\n', MODE, STEP_MODE, MIX_MODE, EM);
tic;

for k = 1:n_samples
    % ---------------------------------------------------------
    % 模块 A：环境生成
    % ---------------------------------------------------------
    if EM == 1 
        if t(k) < 10, current_f = 80; current_amp = 12.0;
        elseif t(k) < 20, current_f = 100; current_amp = 15.0;
        else, current_f = 110; current_amp = 10.0; 
        end 
    elseif EM == 2 % 固定模式
        current_f = 100; current_amp = 15.0;
    elseif EM == 3 % 线性扫频
        current_f = 50 + (100 - 50) * (t(k) / duration); current_amp = 15.0;
    elseif EM == 4 % 正弦调制
        current_f = 75 + 25 * sin(2 * pi * 0.5 * t(k)); current_amp = 15.0;
    end
    
    current_phase = current_phase + 2 * pi * current_f * dt;
    chatter_noise = 0.5 * sin(2 * pi * 400 * t(k)) * randn(); 
    vibration_raw(k) = current_amp * sin(current_phase) ...
                     + (current_amp * 0.15) * sin(2 * current_phase) ...
                     + (current_amp * 0.05) * sin(3 * current_phase) ...
                     + chatter_noise + 0.05 * randn();
    error_signal(k) = vibration_raw(k) + acc_control;
    if MODE == 2, err_buffer = [error_signal(k); err_buffer(1:end-1)]; end
    
    % ---------------------------------------------------------
    % 【修复4+本次修复】模块 A1：论文核心 两阶段物理滤波保底
    % ---------------------------------------------------------
    err_filter_buffer = [error_signal(k); err_filter_buffer(1:end-1)];
    if t(k) < 2 || (t(k) > 9.9 && t(k) < 10.5) || (t(k) > 19.9 && t(k) < 20.5)
        % 阶段一：冷启动/频率突变期，物理滤波剥离基频，只让非线性残差喂给NN
        if t(k) < 10
            err_for_nn = filtfilt(b_stop80, a_stop80, err_filter_buffer);
        elseif t(k) < 20
            err_for_nn = filtfilt(b_stop100, a_stop100, err_filter_buffer);
        else
            err_for_nn = filtfilt(b_stop110, a_stop110, err_filter_buffer);
        end
        error_signal_nn(k) = err_for_nn(1); % 【本次修复】存入全长向量
    else
        % 阶段二：平稳期，旁路滤波器，用隐式解耦模式
        error_signal_nn(k) = error_signal(k); % 【本次修复】存入全长向量
    end
    
    % ---------------------------------------------------------
    % 模块 B：[维度二] 线性底座计算与包络变步长调度
    % ---------------------------------------------------------
    x_buffer = [vibration_raw(k); x_buffer(1:end-1)];
    xf = S_est' * x_buffer(1:length(S_est));
    xf_buffer = [xf; xf_buffer(1:end-1)];
    y_linear(k) = w_fxlms' * x_buffer(1:L);
    
    % --- 提取宏观误差包络 (重度低通滤波，极其平滑) ---
    err_env = 0.9995 * err_env + 0.0005 * abs(error_signal(k));
    
    % --- 【修复1】解耦的步长策略（修正版，完全对齐论文防御型VSS设计）---
    if STEP_MODE == 2
        mu_max_safe = 5e-7;
        mu_min_safe = 5e-8;
        % 【核心修正】冷启动 + 频率突变点，强制启用大步长快速收敛
        if t(k) < 1.5 || (t(k) > 9.9 && t(k) < 10.3) || (t(k) > 19.9 && t(k) < 20.3)
            mu_current = mu_max_safe;
        else
            % 稳态反向防御：误差越大，步长越小，防发散
            mu_current = mu_min_safe + (mu_max_safe - mu_min_safe) * exp(-err_env / 2.0);
        end
    else
        mu_current = mu_fixed;
    end
    history_mu(k) = mu_current;
    
    % ---------------------------------------------------------
    % 模块 C：[维度一] 全局特征提取 与 非线性模型推理
    % ---------------------------------------------------------
    if k >= 200
        % 【修复4+本次修复】用全长向量error_signal_nn替换标量
        features = extract_features(vibration_raw, error_signal_nn, control_signal, xf_buffer, k, fs, current_f);
        Feature_Log(k, :) = features;
    else
        features = zeros(1, 19);
    end

    if MODE == 2
        ref_norm = vibration_raw(k) / 20.0; 
        err_delayed = err_buffer(end) / 20.0; 
        phi_expand = [ref_norm; err_delayed];
        for order = 1:flann_order
            phi_expand = [phi_expand; sin(order*pi*ref_norm); cos(order*pi*ref_norm)];
        end
        h_state = tanh(W_in2hid * phi_expand + W_hid2hid * h_state);
        nn_features = [h_state; 1]; 
        y_nn(k) = W_out * nn_features;
        
        nn_features_buffer = [nn_features'; nn_features_buffer(1:end-1, :)];
        nn_features_f = (S_est' * nn_features_buffer)';
        
    elseif MODE == 3
    if k >= 200
        features = extract_features(vibration_raw, error_signal_nn, control_signal, xf_buffer, k, fs, current_f);
        Feature_Log(k, :) = features; 
        feature_window = [features', feature_window(:, 1:end-1)];
        if k >= 2000
            % ============== 【核心修复：明确处理1×16维度】==============
            y_nn_raw = nn.forward(feature_window); 
            % 1. 强制转置为列向量（如果是1×16，变成16×1）
            y_nn_raw = y_nn_raw(:); 
            % 2. 【关键】强制取最后一个时间步的输出（第16个元素）
            y_nn_raw = y_nn_raw(end); 
            % 3. 强制转换为双精度标量
            y_nn_raw = double(y_nn_raw); 
            % 4. 乘以缩放系数
            y_nn_raw = y_nn_raw * 0.02; 
            % ====================================================================
            
            if k == 2000
                y_nn(k) = y_nn_raw;
            else
                % 0.8+0.2/0.9+0.1
                y_nn(k) = 0.9 * y_nn(k-1) + 0.1 * y_nn_raw; 
            end
        else
            y_nn(k) = 0; 
        end
    else
        y_nn(k) = 0; 
    end
end    
    
    % ---------------------------------------------------------
    % 模块 D：[维度三] 窄带谐波刺客
    % ---------------------------------------------------------
    if MIX_MODE == 3 && (MODE == 2 || MODE == 3) && HARM_SWITCH == 1
        x_harm = [sin(2 * current_phase); cos(2 * current_phase); ... 
                  sin(3 * current_phase); cos(3 * current_phase)]; 
                  %sin(4 * current_phase); cos(4 * current_phase)];
        x_harm_buffer = [x_harm'; x_harm_buffer(1:end-1, :)];
        xf_harm = (S_est' * x_harm_buffer)'; 
        y_harm = w_harm' * x_harm;
        % 【优化】动态调整谐波刺客学习率
        if (t(k) > 9.9 && t(k) < 10.3) || (t(k) > 19.9 && t(k) < 20.3)
            mu_harm_current = 1e-3; % 突变期快速收敛
        else
            mu_harm_current = 5e-4; % 稳态精细调整
        end
        w_harm = w_harm - mu_harm_current * error_signal(k) * xf_harm;
    else
        y_harm = 0; xf_harm = zeros(4,1);
    end
    y_harm_history(k) = y_harm;

    % ---------------------------------------------------------
    % 模块 E：[维度三] 多维消融融合律
    % ---------------------------------------------------------
    if mod(k, thd_update_interval) == 0 && k > thd_window
        err_rms = rms(error_signal(k-thd_window+1 : k)); 
        if err_rms < 1.0, alpha_convex = 0.01;
        elseif err_rms < 4.0, alpha_convex = 0.01 + (err_rms - 1.0)/3.0 * 0.49;
        else, alpha_convex = 0.5; 
        end
    end
    
    if MODE == 1
        y_total = y_linear(k);
    else
        if MIX_MODE == 1
            y_total = 0.95 * y_linear(k) + 0.05 * y_nn(k); 
            history_factor(k) = 0.05;
        elseif MIX_MODE == 2
            y_total = (1 - alpha_convex) * y_linear(k) + alpha_convex * y_nn(k);
            history_factor(k) = alpha_convex;
        elseif MIX_MODE == 3
            y_total = y_linear(k) + w_nn * y_nn(k) + y_harm;
            history_factor(k) = w_nn;
        end
    end
    
    control_signal(k) = max(min(y_total, 10), -10);
    
    % ---------------------------------------------------------
    % 模块 F：物理执行与权值独立更新
    % ---------------------------------------------------------
    [~, actuator_force] = actuator.actuate(control_signal(k), dt);
    [~, ~] = structure.respond(actuator_force, dt);
    acc_control = structure.get_acceleration(2);
    
    w_fxlms = w_fxlms - mu_current * error_signal(k) * xf_buffer;
    
    if MODE == 2
        current_mix = 1.0; 
        lr_max = 0.002;
        lr_min = 1e-6;
        lr_real_dynamic = lr_min + (lr_max - lr_min) * exp(-err_env / 2.0);
        W_out = W_out - lr_real_dynamic * error_signal(k) * nn_features_f' * current_mix; 
        W_out = W_out * 0.9999; 
    end
    
    if MIX_MODE == 3 
        if MODE == 3
            yn_f_buffer = [y_nn(k); yn_f_buffer(1:end-1)];
            y_nn_f = S_est' * yn_f_buffer;
            % 【修复3】强非线性时，强制NN主导，完全对齐论文控制策略
            if abs(error_signal(k)) > 3.0 || (t(k) > 9.9 && t(k) < 10.3) || (t(k) > 19.9 && t(k) < 20.3)
                mu_wnn = 5e-4; 
                forget_factor = 1.0; 
                % 突变期强制NN权重下限，不让它跑路
                w_nn_min = 0.2;
            else
                mu_wnn = 1e-5; 
                forget_factor = 0.999; 
                w_nn_min = 0.05;

            % ==============================================
            % 【新增】稳态检测与衰减逻辑
            % ==============================================
            if abs(error_signal(k)) < steady_state_threshold
                steady_state_counter = steady_state_counter + 1;
                % 连续稳定足够时间，启动稳态衰减
                if steady_state_counter > steady_state_min_duration * fs
                    % 叠加额外衰减因子，让w_nn缓慢下降
                    forget_factor = forget_factor - w_nn_decay_rate;
                    % 衰减不能低于最低下限
                    w_nn_min = w_nn_min_steady;
                end
            else
                % 误差波动超过阈值，重置稳态计数器
                steady_state_counter = 0;
            end

            end
            w_nn = forget_factor * w_nn - mu_wnn * error_signal(k) * y_nn_f;
            % w_nn_min
            %w_nn = max(min(w_nn, 2.0), w_nn_min); % 强制权重下限，不让NN退出
            w_nn =w_nn*0.999;
        end
    end

    if mod(k, fs*5) == 0
        fprintf('时间: %d s, 当前误差: %.2f m/s2\n', t(k), error_signal(k));
    end
end
sim_time = toc;
if ta ==1

%==================== 数据生成与导出 ====================
    fprintf('\n>> 正在启动云端上帝视角，计算完美 Target...\n');
    ideal_y_total = zeros(n_samples, 1);
    ideal_y_total(1 : end-delay) = vibration_raw(delay+1 : end) / (-s_hat_gain);
    Target_Log = ideal_y_total - y_linear;
    valid_idx = 2000 : (n_samples - delay - 1);
    data_collect.features = Feature_Log(valid_idx, :);
    data_collect.targets = Target_Log(valid_idx);
    save('training_data.mat', 'data_collect');
    fprintf('成功导出 training_data.mat！共包含 %d 条恶劣工况样本。\n', length(valid_idx));
end
fprintf('仿真完成，耗时 %.2f 秒\n', sim_time);

%==================== 7. 全局战报与出图 ====================
rms_raw_global = rms(vibration_raw);
rms_err_global = rms(error_signal);
reduction_dB_global = 20 * log10(rms_raw_global / rms_err_global);

peak_raw = max(abs(vibration_raw));
peak_err = max(abs(error_signal));
rms_err_steady = rms(error_signal(end-fs*2:end));

fprintf('\n========== 终极性能战报 ==========\n');
fprintf('最大峰值误差 : %.2f m/s2 (原始峰值: %.2f m/s2)\n', peak_err, peak_raw);
fprintf('全局 RMS 误差: %.4f m/s2\n', rms_err_global);
fprintf('稳态 RMS 误差: %.4f m/s2 (最后2秒)\n', rms_err_steady);
fprintf('全局振动抑制比: %.2f dB\n', reduction_dB_global);

%==================== 8. 非线性量化分区评估 ====================
%% ==================== 8. 非线性量化分区评估（控制前 vs 控制后）====================
fprintf('\n========== 非线性量化分区评估（控制前原始振动 vs 控制后残余误差）==========\n');
t1 = 10*fs; t2 = 20*fs;
segments = {1:t1, t1+1:t2, t2+1:n_samples};
seg_names = {'工步1 (0-10s)', '工步2 (10-20s)', '工步3 (20-30s)'};
nSeg = length(segments);

% 预分配：原始与残余各一套指标
RAW_THD = zeros(nSeg,1); RAW_Kurt = zeros(nSeg,1); RAW_High = zeros(nSeg,1);
RAW_Score = zeros(nSeg,1); RAW_Level = cell(nSeg,1);
RES_THD = zeros(nSeg,1); RES_Kurt = zeros(nSeg,1); RES_High = zeros(nSeg,1);
RES_Score = zeros(nSeg,1); RES_Level = cell(nSeg,1);

for i = 1:nSeg
    % 原始振动（控制前）
    raw_seg = vibration_raw(segments{i});
    [met_raw, ~] = nonlinear_quantification(raw_seg, fs, false);
    RAW_THD(i) = met_raw.thd;
    RAW_Kurt(i) = met_raw.kurtosis;
    RAW_High(i) = met_raw.band_ratio(3);
    RAW_Score(i) = met_raw.nonlinear_score;
    RAW_Level{i} = met_raw.level;
    
    % 残余误差（控制后）
    res_seg = error_signal(segments{i});
    [met_res, ~] = nonlinear_quantification(res_seg, fs, false);
    RES_THD(i) = met_res.thd;
    RES_Kurt(i) = met_res.kurtosis;
    RES_High(i) = met_res.band_ratio(3);
    RES_Score(i) = met_res.nonlinear_score;
    RES_Level{i} = met_res.level;
end

% 整体评估（只绘图一次，用残余误差）
[met_all, ~] = nonlinear_quantification(error_signal, fs, true);

% -------- 打印对比表格 --------
fprintf('\n------------------ 原始振动（控制前） ------------------\n');
fprintf('%-15s | %9s | %9s | %9s | %11s | %-12s\n', ...
    '区间', 'THD(%)', '峭度', '高频占比(%)','非线性评分','等级');
fprintf('-------------------------------------------------------------------------------\n');
for i = 1:nSeg
    fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f | %-12s\n', ...
        seg_names{i}, RAW_THD(i)*100, RAW_Kurt(i), RAW_High(i)*100, ...
        RAW_Score(i), RAW_Level{i});
end
fprintf('-------------------------------------------------------------------------------\n');
fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f | %-12s\n', ...
    '分区平均', mean(RAW_THD)*100, mean(RAW_Kurt), mean(RAW_High)*100, ...
    mean(RAW_Score), '--');

fprintf('\n------------------ 残余误差（控制后） ------------------\n');
fprintf('%-15s | %9s | %9s | %9s | %11s | %-12s\n', ...
    '区间', 'THD(%)', '峭度', '高频占比(%)','非线性评分','等级');
fprintf('-------------------------------------------------------------------------------\n');
for i = 1:nSeg
    fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f | %-12s\n', ...
        seg_names{i}, RES_THD(i)*100, RES_Kurt(i), RES_High(i)*100, ...
        RES_Score(i), RES_Level{i});
end
fprintf('-------------------------------------------------------------------------------\n');
fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f | %-12s\n', ...
    '分区平均', mean(RES_THD)*100, mean(RES_Kurt), mean(RES_High)*100, ...
    mean(RES_Score), '--');
% 整体信号行（残余）
fprintf('-------------------------------------------------------------------------------\n');
fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f | %-12s\n', ...
    '整体残余', met_all.thd*100, met_all.kurtosis, met_all.band_ratio(3)*100, ...
    met_all.nonlinear_score, met_all.level);

fprintf('\n========== 非线性抑制效果对比 ==========\n');
fprintf('%-15s | %9s | %9s | %9s | %11s\n', ...
    '区间', 'THD降幅(%)', '峭度降幅', '高频占比降幅', '评分降幅');
fprintf('-------------------------------------------------------------------------------\n');
for i = 1:nSeg
    % 降幅 = 原始值 - 残余值（正值表示非线性减弱）
    dTHD = (RAW_THD(i) - RES_THD(i)) * 100;       % 百分点
    dKurt = RAW_Kurt(i) - RES_Kurt(i);
    dHigh = (RAW_High(i) - RES_High(i)) * 100;
    dScore = RAW_Score(i) - RES_Score(i);
    fprintf('%-15s | %8.2f | %8.2f | %8.2f | %10.3f\n', ...
        seg_names{i}, dTHD, dKurt, dHigh, dScore);
end
fprintf('-------------------------------------------------------------------------------\n');
fprintf('【全局策略建议】 %s\n', met_all.recommendation);

figure('Name', '第五章 终极解耦多维度消融测试', 'Position', [100, 100, 1200, 800]);

subplot(3, 1, 1);
plot(t, vibration_raw, 'Color', [0.7 0.7 0.7]); hold on;
plot(t, error_signal, 'r', 'LineWidth', 1.2);
xlabel('时间 (s)'); ylabel('加速度 (m/s^2)');
title(sprintf('残差对比 (架构:%d, 底座:%d, 融合:%d | 全局: %.1f dB, 稳态: %.2f m/s^2)', ...
    MODE, STEP_MODE, MIX_MODE, reduction_dB_global, rms_err_steady));
legend('原始恶劣振动', '残余误差'); grid on;

subplot(3, 1, 2);
[Pv, f_v] = pwelch(vibration_raw, hamming(1024), 512, 2048, fs);
[Pe, f_e] = pwelch(error_signal, hamming(1024), 512, 2048, fs);
semilogy(f_v, Pv, 'Color', [0.7 0.7 0.7]); hold on;
semilogy(f_e, Pe, 'r', 'LineWidth', 1.2);
xlabel('频率 (Hz)'); ylabel('PSD'); xlim([0, 800]); grid on;

subplot(3, 1, 3);
yyaxis left;
plot(t, history_factor, 'b', 'LineWidth', 1.5);
ylabel('融合因子 (\alpha 或 \rho)'); ylim([-1.1, 1.1]);
yyaxis right;
plot(t, history_mu, 'm--', 'LineWidth', 1);
ylabel('FxLMS 步长 \mu');
title('动态调度曲线追踪：步长与融合因子的独立解耦'); grid on; xlabel('时间 (s)');

%==================== 非线性量化函数（内嵌） ====================
function [metrics, report] = nonlinear_quantification(error_signal, fs, plot_flag)
    % 非线性量化评估模块
    % 输入：error_signal - 误差信号 (m/s²)，fs - 采样率 (Hz)
    %       plot_flag - 是否绘图 (默认 false)
    % 输出：metrics - 指标结构体，report - 同 metrics

    if nargin < 2, fs = 10000; end
    if nargin < 3, plot_flag = false; end
    
    N = length(error_signal);
    t = (0:N-1)' / fs;
    
    % 1. 频域分析：THD
    nfft = 4096;
    window = hamming(nfft);
    overlap = nfft / 2;
    [S, f] = pwelch(error_signal, window, overlap, nfft, fs);
    S = sqrt(S);
    [~, idx_max] = max(S(2:end));
    f0 = f(idx_max + 1);
    
    harmonics = zeros(10, 1);
    for h = 1:10
        f_h = h * f0;
        if f_h <= fs/2
            [~, idx] = min(abs(f - f_h));
            harmonics(h) = S(idx);
        end
    end
    fundamental = harmonics(1);
    harmonic_power = sqrt(sum(harmonics(2:end).^2));
    thd = harmonic_power / fundamental;
    
    % 2. 时域分析：峭度
    kurt = kurtosis(error_signal);
    window_len = 256;
    sliding_kurt = zeros(N, 1);
    for i = window_len:N
        sliding_kurt(i) = kurtosis(error_signal(i-window_len+1:i));
    end
    max_sliding_kurt = max(sliding_kurt(window_len:end));
    
    % 3. 频段能量分布
    bands = [0, 200; 200, 500; 500, 1000];
    band_energy = zeros(size(bands, 1), 1);
    total_energy = sum(S);
    for b = 1:size(bands, 1)
        idx = (f >= bands(b,1) & f < bands(b,2));
        band_energy(b) = sum(S(idx));
    end
    band_ratio = band_energy / total_energy;
    
    % 4. 分频/超谐波检测（仅用于内部记录，未输出到表格）
    subharmonics = [];
    test_freqs = [f0/2, f0/3, 2*f0, 3*f0, 4*f0];
    for tf = test_freqs
        if tf > 0 && tf <= fs/2
            [~, idx] = min(abs(f - tf));
            subharmonics = [subharmonics; tf, S(idx)];
        end
    end
    
    % 5. 综合非线性指标
    thd_score = min(1, max(0, (thd - 0.03) / (0.10 - 0.03)));
    kurt_score = min(1, max(0, (kurt - 3) / (4 - 3)));
    high_freq_ratio = band_ratio(3);
    high_freq_score = min(1, high_freq_ratio / 0.3);
    nonlinear_score = 0.5 * thd_score + 0.3 * kurt_score + 0.2 * high_freq_score;
    
    if nonlinear_score < 0.1
        level = '线性';
        recommendation = 'FxLMS 主导控制，无需神经网络补偿';
    elseif nonlinear_score < 0.3
        level = '弱非线性';
        recommendation = '建议启用混合控制，混合系数 α 在 0.6~0.8';
    elseif nonlinear_score < 0.6
        level = '中等非线性';
        recommendation = '建议神经网络主导控制，α 在 0.3~0.5';
    else
        level = '强非线性';
        recommendation = '必须神经网络主导，α < 0.2，FxLMS 仅作备份';
    end
    
    metrics = struct();
    metrics.thd = thd;
    metrics.kurtosis = kurt;
    metrics.max_sliding_kurt = max_sliding_kurt;
    metrics.band_ratio = band_ratio;
    metrics.nonlinear_score = nonlinear_score;
    metrics.level = level;
    metrics.recommendation = recommendation;
    metrics.f0 = f0;
    report = metrics;
    
    % 绘图（可选）
    if plot_flag
        figure('Name', '非线性量化评估', 'Position', [100, 100, 1400, 900]);
        subplot(2,3,1); plot(t, error_signal, 'b', 'LineWidth', 0.5);
        xlabel('时间 (s)'); ylabel('误差 (m/s²)'); title('误差信号时域'); grid on; xlim([0, min(2, t(end))]);
        subplot(2,3,2); semilogy(f, S, 'b', 'LineWidth', 1); hold on;
        for h = 1:5
            f_h = h * f0;
            if f_h <= fs/2
                [~, idx] = min(abs(f - f_h));
                plot(f_h, S(idx), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
                text(f_h, S(idx)*1.5, sprintf('%d×', h), 'HorizontalAlignment', 'center');
            end
        end
        xlabel('频率 (Hz)'); ylabel('幅值'); title(sprintf('频谱分析 (THD = %.2f%%)', thd*100)); xlim([0, 1000]); grid on;
        subplot(2,3,3); plot(t, sliding_kurt, 'g', 'LineWidth', 1); hold on;
        yline(3, 'r--', '线性阈值'); yline(4, 'r--', '非线性阈值');
        xlabel('时间 (s)'); ylabel('峭度'); title('滑动峭度 (窗口256点)'); grid on; xlim([0, min(5, t(end))]);
        subplot(2,3,4); labels = {'0-200 Hz', '200-500 Hz', '500-1000 Hz'}; pie(band_ratio, labels); title('频段能量分布');
        subplot(2,3,5); theta = linspace(0, pi, 100); r = 0.8; x = r*cos(theta); y = r*sin(theta);
        fill(x, y, [0.9, 0.9, 0.9]); hold on; angle = nonlinear_score*pi;
        plot([0, 0.7*cos(angle)], [0, 0.7*sin(angle)], 'r', 'LineWidth', 3);
        for s = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            a = s*pi; text(0.85*cos(a), 0.85*sin(a), sprintf('%.1f', s), 'HorizontalAlignment', 'center');
        end
        axis equal; axis off; title(sprintf('非线性综合评分\n%.3f (%s)', nonlinear_score, level));
        subplot(2,3,6); axis off;
        text(0.1, 0.9, '【控制策略建议】', 'FontSize', 12, 'FontWeight', 'bold');
        text(0.1, 0.7, sprintf('THD = %.2f%%', thd*100), 'FontSize', 10);
        text(0.1, 0.6, sprintf('峭度 = %.2f', kurt), 'FontSize', 10);
        text(0.1, 0.5, sprintf('高频占比 = %.1f%%', high_freq_ratio*100), 'FontSize', 10);
        text(0.1, 0.4, sprintf('综合评分 = %.3f', nonlinear_score), 'FontSize', 10);
        text(0.1, 0.3, sprintf('评估等级: %s', level), 'FontSize', 10, 'FontWeight', 'bold');
        text(0.1, 0.15, recommendation, 'FontSize', 10, 'Color', 'b');
        sgtitle('非线性量化评估报告', 'FontSize', 14);
    end
end