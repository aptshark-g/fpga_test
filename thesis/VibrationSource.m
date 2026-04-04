% 振动源模型类
classdef VibrationSource < handle
    properties
        % 振动特性参数 [基于王加春(2001)实测数据]
        frequencies = [50, 250, 800];   % 振动频率数组 [Hz]
        amplitudes = [3.2, 2.8, 1.5];   % 对应幅值数组 [m/s²]
        phases = [0, pi/4, pi/2];       % 相位数组 [rad]
        noise_level = 0.05;             % 噪声水平
        shock_probability = 0.01;       % 随机冲击发生概率
        
        % 双频振动特性参数 [借鉴顶刊[10]双频激励模式]
        dual_frequencies = [50, 250];   % 双频组合 [Hz]
        dual_amplitude_ratio = 2/3;     % 幅值比 (A2/A1) = 2/3，即3:2 (A1:A2)
        rpm_frequency_coefficient = 0.5e-3; % 转速-频率漂移系数 [Hz/rpm]
    end
    
    methods
        function signal = generate_dual_frequency(obj, t, condition)
            % 生成双频耦合振动信号
            
            % 初始化信号
            signal = zeros(size(t));
            
            % 模拟主轴转速波动引起的频率调制效应
            spindle_rpm = condition.spindle_rpm; % 假设已经验证过该字段存在
            
            % 计算调制后的双频频率
            f1 = obj.dual_frequencies(1) + obj.rpm_frequency_coefficient * spindle_rpm;
            f2 = obj.dual_frequencies(2) + obj.rpm_frequency_coefficient * spindle_rpm;
            
            % 计算幅值（根据幅值比3:2，即A1:A2=3:2，A2/A1=2/3）
            A1 = obj.amplitudes(1);  % 主频幅值 (50Hz)
            A2 = A1 * obj.dual_amplitude_ratio;  % 次频幅值 (250Hz)，A2 = A1 * 2/3
            
            % 生成双频耦合信号
            phase_diff = pi/3;  % 60度相位差
            signal = A1 * sin(2*pi*f1*t) + A2 * sin(2*pi*f2*t + phase_diff);
            
            % 添加幅值调制
            if condition.enable_amplitude_modulation
                amp_mod = 0.15 * sin(2*pi*0.3*t);
                signal = signal .* (1 + amp_mod);
            end
            
            % 添加测量噪声
            if condition.enable_noise
                sig_rms = rms(signal);
                if sig_rms == 0, sig_rms = 1; end
                noise_power = obj.noise_level * sig_rms;
                signal = signal + noise_power * randn(size(signal));
            end
        fprintf('双频振动: %.0f Hz (%.2f m/s²) + %.0f Hz (%.2f m/s²), 幅值比 %.1f:%.1f\n',f1, A1, f2, A2, 1, obj.dual_amplitude_ratio);
        end

		function [freq_spectrum, f] = analyze_spectrum(obj, t, signal)
		% 频谱分析工具方法
		L = length(signal);
		dt = t(2) - t(1);
		fs = 1/dt;
		
		f = fs * (0:(L/2)) / L;
		Y = fft(signal);
		P2 = abs(Y/L);
		freq_spectrum = P2(1:floor(L/2)+1);
		freq_spectrum(2:end-1) = 2*freq_spectrum(2:end-1);
		end
		
		function plot_dual_frequency_comparison(obj, t, condition)
		% 绘制单频与双频振动的对比图
		% 确保输入condition有效
		if nargin < 3, condition = struct(); end
		condition = obj.validate_condition(condition);
		
		% 生成单频振动信号
		condition1 = condition;
		condition1.dual_frequency_mode = false;
		signal_single = obj.generate(t, condition1);
		
		% 生成双频振动信号
		condition2 = condition;
		condition2.dual_frequency_mode = true;
		signal_dual = obj.generate(t, condition2);
		
		% 绘制时域对比
		figure('Name', '振动模式对比');
		subplot(2,1,1);
		plot(t, signal_single, 'b', 'LineWidth', 1);
		title('单频振动信号 (基准)');
		xlabel('时间 (s)'); ylabel('加速度 (m/s²)');
		grid on;
		
		subplot(2,1,2);
		plot(t, signal_dual, 'r', 'LineWidth', 1);
		title('双频耦合振动信号 (顶刊工况)');
		xlabel('时间 (s)'); ylabel('加速度 (m/s²)');
		grid on;
		
		% 绘制频域对比
		[P1, f1] = obj.analyze_spectrum(t, signal_single);
		[P2, f2] = obj.analyze_spectrum(t, signal_dual);
		
		figure('Name', '频谱对比');
		subplot(2,1,1);
		plot(f1, P1, 'b', 'LineWidth', 1.5);
		title('单频振动频谱');
		xlabel('频率 (Hz)'); ylabel('幅值');
		grid on;
		xlim([0, 1000]);
		
		subplot(2,1,2);
		plot(f2, P2, 'r', 'LineWidth', 1.5);
		title('双频耦合振动频谱');
		xlabel('频率 (Hz)'); ylabel('幅值');
		grid on;
		xlim([0, 1000]);
		end
		
		% === 辅助函数：参数校验 ===
		function cond_out = validate_condition(obj, cond_in)
		% 填充缺失的字段，防止报错
		cond_out = cond_in;
		if ~isfield(cond_out, 'dual_frequency_mode'), cond_out.dual_frequency_mode = false; end
		if ~isfield(cond_out, 'enable_frequency_modulation'), cond_out.enable_frequency_modulation = false; end
		if ~isfield(cond_out, 'enable_amplitude_modulation'), cond_out.enable_amplitude_modulation = false; end
		if ~isfield(cond_out, 'enable_random_shock'), cond_out.enable_random_shock = false; end
		if ~isfield(cond_out, 'enable_noise'), cond_out.enable_noise = false; end
		if ~isfield(cond_out, 'spindle_rpm'), cond_out.spindle_rpm = 3000; end % 默认转速
		if ~isfield(cond_out, 'sample_rate'), cond_out.sample_rate = 10000; end
		end
		end
		end