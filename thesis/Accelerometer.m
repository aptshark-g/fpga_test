% 传感器模型类
classdef Accelerometer < handle
    properties
        % 基本参数 [基于曾威(2006)所用PCB 352C33]
        sensitivity = 98;                  % 灵敏度 [mV/g]
        bandwidth = [0.5, 3000];           % 带宽 [Hz]
        noise_density = 6e-6;              % 噪声密度 [g/√Hz] (6 μg/√Hz)
        full_scale_range = 50;             % 满量程 [g]
        
        % 采样参数
        sample_rate = 10000;               % 采样率 [Hz]
        bit_resolution = 16;               % ADC位数
        quantization_error = 0;             % 量化误差标志
        
        % 滤波器系数
        filter_coeff_b = [];               % 滤波器分子系数
        filter_coeff_a = [];               % 滤波器分母系数
        
        % 校准参数
        calibration_factor = 1.0;          % 校准系数
        offset = 0;                        % 零点偏移 [V]
        
        % 状态变量
        filter_state = [];                 % 滤波器状态
        prev_sample = 0;                   % 前一采样值
    end
    
    methods
        function obj = Accelerometer(sample_rate, bandwidth)
            % 传感器构造函数
            % 输入: 
            %   sample_rate - 采样率 [Hz]
            %   bandwidth - 带宽 [Hz] (可选)
            
            if nargin >= 1
                obj.sample_rate = sample_rate;
            end
            if nargin >= 2
                obj.bandwidth = bandwidth;
            end
            
            % 设置参数（根据表格）
            obj.sensitivity = 98;          % 灵敏度 98 mV/g
            obj.noise_density = 6e-6;      % 噪声密度 6 μg/√Hz
            obj.bandwidth = [0.5, 3000];   % 带宽 0.5-3000 Hz
            
            % 设计抗混叠滤波器
            obj.design_filter();
            
            % 初始化滤波器状态
            order = max(length(obj.filter_coeff_b), length(obj.filter_coeff_a)) - 1;
            obj.filter_state = zeros(order, 1);
            
            fprintf('加速度计初始化: PCB 352C33\n');
            fprintf('  灵敏度: %.0f mV/g\n', obj.sensitivity);
            fprintf('  带宽: %.1f-%.0f Hz\n', obj.bandwidth(1), obj.bandwidth(2));
            fprintf('  噪声密度: %.1f μg/√Hz\n', obj.noise_density*1e6);
        end
        
        function design_filter(obj)
            % 设计抗混叠滤波器（带通滤波器）
            
            fs = obj.sample_rate;
            
            % 设计带通滤波器（巴特沃斯，4阶）
            f_low = obj.bandwidth(1);
            f_high = obj.bandwidth(2);
            
            % 归一化截止频率
            Wn = [f_low, f_high] / (fs/2);
            
            % 确保截止频率在(0,1)范围内
            Wn(Wn <= 0) = 0.001;
            Wn(Wn >= 1) = 0.999;
            
            % 设计带通滤波器
            [obj.filter_coeff_b, obj.filter_coeff_a] = butter(4, Wn, 'bandpass');
        end
        
        function voltage = measure(obj, acceleration, t)
            % 模拟加速度计测量，包含滤波、噪声添加和量化
            % 输入: acceleration - 加速度 [m/s²] 或数组, t - 时间 [s] 或数组
            % 输出: voltage - 输出电压 [V]
            
            % 处理标量输入
            if isscalar(acceleration)
                voltage = obj.measure_single(acceleration, t);
                return;
            end
            
            % 处理数组输入
            n_samples = length(acceleration);
            voltage = zeros(n_samples, 1);
            
            for i = 1:n_samples
                if nargin < 3 || isscalar(t)
                    t_i = i / obj.sample_rate;
                else
                    t_i = t(i);
                end
                voltage(i) = obj.measure_single(acceleration(i), t_i);
            end
        end
        
        function voltage = measure_single(obj, acceleration, t)
            % 单个采样点的测量
            
            % 1. 转换为g单位
            g = 9.80665;  % 标准重力加速度 [m/s²]
            acceleration_g = acceleration / g;
            
            % 2. 应用带宽限制（滤波）
            if ~isempty(obj.filter_coeff_b)
                [acceleration_g, obj.filter_state] = filter(...
                    obj.filter_coeff_b, obj.filter_coeff_a, ...
                    acceleration_g, obj.filter_state);
            end
            
            % 3. 添加传感器噪声
            if obj.noise_density > 0
                % 计算噪声标准差
                noise_bandwidth = obj.bandwidth(2) - obj.bandwidth(1);
                noise_std = obj.noise_density * sqrt(noise_bandwidth);
                
                % 添加高斯白噪声
                noise = noise_std * randn();
                acceleration_g = acceleration_g + noise;
            end
            
            % 4. 转换为电压（考虑灵敏度）
            voltage_mv = acceleration_g * obj.sensitivity;  % [mV]
            voltage = voltage_mv / 1000;                    % [V]
            
            % 5. 应用校准和偏移
            voltage = voltage * obj.calibration_factor + obj.offset;
            
            % 6. 量程限制
            max_voltage = obj.sensitivity * obj.full_scale_range / 1000;
            voltage = max(min(voltage, max_voltage), -max_voltage);
            
            % 7. 量化（如果启用）
            if obj.quantization_error && obj.bit_resolution > 0
                voltage = obj.apply_quantization(voltage, max_voltage);
            end
            
            % 更新前一采样值
            obj.prev_sample = voltage;
        end
        
        function quantized_value = apply_quantization(obj, value, max_value)
            % 应用量化误差
            % 输入: value - 原始值, max_value - 满量程值
            % 输出: quantized_value - 量化后的值
            
            % 计算量化步长
            quantization_step = 2 * max_value / (2^obj.bit_resolution - 1);
            
            % 量化
            quantized_value = round(value / quantization_step) * quantization_step;
        end
        
        function calibrate(obj, known_acceleration, measured_voltage)
            % 传感器校准
            % 输入: known_acceleration - 已知加速度 [m/s²]
            %       measured_voltage - 测得的电压 [V]
            
            g = 9.80665;
            acceleration_g = known_acceleration / g;
            
            expected_voltage = acceleration_g * obj.sensitivity / 1000;
            
            % 计算校准系数
            if measured_voltage ~= 0
                obj.calibration_factor = expected_voltage / measured_voltage;
            end
            
            % 计算零点偏移
            obj.offset = measured_voltage - expected_voltage * obj.calibration_factor;
            
            fprintf('校准完成: 校准系数=%.4f, 零点偏移=%.6f V\n', ...
                obj.calibration_factor, obj.offset);
        end
        
        function reset(obj)
            % 重置传感器状态
            order = max(length(obj.filter_coeff_b), length(obj.filter_coeff_a)) - 1;
            obj.filter_state = zeros(order, 1);
            obj.prev_sample = 0;
        end
        
        function plot_frequency_response(obj)
            % 绘制传感器频率响应
            
            fs = obj.sample_rate;
            n_points = 1000;
            freqs = logspace(0, log10(fs/2), n_points);
            
            % 计算滤波器响应
            [H, f] = freqz(obj.filter_coeff_b, obj.filter_coeff_a, freqs, fs);
            
            % 计算灵敏度响应
            sensitivity_response = abs(H) * obj.sensitivity / 1000;  % [V/g]
            
            figure('Name', '传感器频率响应', 'Position', [100, 100, 800, 600]);
            
            subplot(2,1,1);
            semilogx(f, 20*log10(sensitivity_response), 'b-', 'LineWidth', 2);
            hold on;
            
            % 标记带宽
            line([obj.bandwidth(1), obj.bandwidth(1)], ylim, ...
                'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
            line([obj.bandwidth(2), obj.bandwidth(2)], ylim, ...
                'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
            
            title('传感器幅频响应');
            xlabel('频率 [Hz]');
            ylabel('增益 [dB]');
            grid on;
            xlim([1, fs/2]);
            legend('幅频响应', '带宽限制', 'Location', 'best');
            
            subplot(2,1,2);
            semilogx(f, angle(H)*180/pi, 'b-', 'LineWidth', 2);
            title('传感器相频响应');
            xlabel('频率 [Hz]');
            ylabel('相位 [度]');
            grid on;
            xlim([1, fs/2]);
            
            sgtitle(sprintf('加速度计频率响应 (PCB 352C33, %.0f mV/g)', obj.sensitivity));
        end
        
        function plot_noise_characteristics(obj, duration)
            % 绘制传感器噪声特性
            % 输入: duration - 持续时间 [s] (可选)
            
            if nargin < 2
                duration = 1;  % 1秒
            end
            
            n_samples = round(duration * obj.sample_rate);
            t = (0:n_samples-1)' / obj.sample_rate;
            
            % 生成纯噪声信号
            noise_only = zeros(n_samples, 1);
            for i = 1:n_samples
                noise_only(i) = obj.measure_single(0, t(i));
            end
            
            % 转换为加速度单位
            g = 9.80665;
            noise_g = noise_only * 1000 / obj.sensitivity;  % [g]
            noise_ms2 = noise_g * g;                        % [m/s²]
            
            figure('Name', '传感器噪声特性', 'Position', [100, 100, 1000, 800]);
            
            % 时域噪声
            subplot(3,2,1);
            plot(t, noise_ms2, 'b-', 'LineWidth', 0.5);
            title('时域噪声');
            xlabel('时间 [s]');
            ylabel('加速度 [m/s²]');
            grid on;
            
            subplot(3,2,2);
            histogram(noise_ms2, 50, 'FaceColor', 'b', 'EdgeColor', 'none');
            title('噪声分布');
            xlabel('加速度 [m/s²]');
            ylabel('频次');
            grid on;
            
            % 频域分析
            subplot(3,2,3);
            [pxx, f] = pwelch(noise_ms2, hamming(512), 256, 512, obj.sample_rate);
            loglog(f, sqrt(pxx), 'b-', 'LineWidth', 1.5);
            hold on;
            
            % 理论噪声密度线
            theoretical_noise = obj.noise_density * g;  % [m/s²/√Hz]
            line(f, theoretical_noise * ones(size(f)), ...
                'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
            
            title('噪声功率谱密度');
            xlabel('频率 [Hz]');
            ylabel('PSD [m/s²/√Hz]');
            grid on;
            legend('实测噪声', '理论噪声密度', 'Location', 'best');
            xlim([obj.bandwidth(1), obj.bandwidth(2)]);
            
            subplot(3,2,4);
            plot(f, 10*log10(pxx), 'b-', 'LineWidth', 1.5);
            title('噪声功率谱 (dB)');
            xlabel('频率 [Hz]');
            ylabel('功率 [dB]');
            grid on;
            xlim([obj.bandwidth(1), obj.bandwidth(2)]);
            
            % 统计信息
            subplot(3,2,5:6);
            axis off;
            
            stats_text = {
                sprintf('传感器型号: PCB 352C33');
                sprintf('灵敏度: %.0f mV/g', obj.sensitivity);
                sprintf('带宽: %.1f-%.0f Hz', obj.bandwidth(1), obj.bandwidth(2));
                sprintf('噪声密度: %.2f μg/√Hz', obj.noise_density*1e6);
                sprintf('满量程: ±%.0f g', obj.full_scale_range);
                sprintf('采样率: %.0f Hz', obj.sample_rate);
                sprintf('---');
                sprintf('噪声统计 (%.1f秒数据):', duration);
                sprintf('  均值: %.3e m/s²', mean(noise_ms2));
                sprintf('  标准差: %.3e m/s²', std(noise_ms2));
                sprintf('  峰峰值: %.3e m/s²', max(noise_ms2)-min(noise_ms2));
                sprintf('  理论噪声: %.3e m/s²', theoretical_noise);
                sprintf('  信噪比: %.1f dB (假设1g信号)', 20*log10(1/std(noise_g)));
            };
            
            text(0.1, 0.5, stats_text, 'FontSize', 10, 'VerticalAlignment', 'middle');
            title('传感器参数与噪声统计');
        end
        
        function test_measurement(obj, test_frequency, test_amplitude, duration)
            % 测试测量功能
            % 输入: test_frequency - 测试频率 [Hz]
            %       test_amplitude - 测试幅值 [m/s²]
            %       duration - 测试持续时间 [s]
            
            if nargin < 4
                duration = 0.1;
            end
            
            n_samples = round(duration * obj.sample_rate);
            t = (0:n_samples-1)' / obj.sample_rate;
            
            % 生成测试信号
            test_signal = test_amplitude * sin(2*pi*test_frequency*t);
            
            % 测量
            measured_voltage = obj.measure(test_signal, t);
            
            % 转换为加速度
            measured_acceleration = measured_voltage * 1000 / obj.sensitivity * 9.80665;
            
            figure('Name', '传感器测量测试', 'Position', [100, 100, 1200, 400]);
            
            subplot(1,3,1);
            plot(t, test_signal, 'b-', 'LineWidth', 1.5);
            hold on;
            plot(t, measured_acceleration, 'r--', 'LineWidth', 1.5);
            title(sprintf('时域对比 (%.0f Hz)', test_frequency));
            xlabel('时间 [s]');
            ylabel('加速度 [m/s²]');
            legend('原始信号', '测量信号', 'Location', 'best');
            grid on;
            
            subplot(1,3,2);
            error = measured_acceleration - test_signal;
            plot(t, error, 'g-', 'LineWidth', 1);
            title('测量误差');
            xlabel('时间 [s]');
            ylabel('误差 [m/s²]');
            grid on;
            
            subplot(1,3,3);
            [f_test, P_test] = periodogram(test_signal, [], [], obj.sample_rate);
            [f_meas, P_meas] = periodogram(measured_acceleration, [], [], obj.sample_rate);
            semilogy(f_test, sqrt(P_test), 'b-', 'LineWidth', 1.5);
            hold on;
            semilogy(f_meas, sqrt(P_meas), 'r--', 'LineWidth', 1.5);
            title('频谱对比');
            xlabel('频率 [Hz]');
            ylabel('幅值 [m/s²]');
            legend('原始信号', '测量信号', 'Location', 'best');
            grid on;
            xlim([0, min(2000, obj.sample_rate/2)]);
            
            % 计算误差统计
            rms_error = rms(error);
            snr = 20*log10(rms(test_signal)/rms_error);
            
            fprintf('\n=== 传感器测试结果 ===\n');
            fprintf('测试频率: %.0f Hz\n', test_frequency);
            fprintf('测试幅值: %.2f m/s² (%.2f g)\n', test_amplitude, test_amplitude/9.80665);
            fprintf('RMS误差: %.3e m/s²\n', rms_error);
            fprintf('信噪比: %.1f dB\n', snr);
            fprintf('相对误差: %.2f%%\n', rms_error/rms(test_signal)*100);
        end
    end
end