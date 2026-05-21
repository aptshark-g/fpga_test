# SDC1.sdc 终极完美版
# ==============================================
# 1. 主时钟与PLL约束（核心时钟定义）
# ==============================================
# 外部输入50MHz参考时钟
create_clock -name sys_clk_in -period 20 -waveform {0 10} [get_ports sys_clk]
# 自动推导PLL输出时钟，继承主时钟约束
derive_pll_clocks -create_base_clocks
# 自动推导时钟不确定性，预留时序余量
derive_clock_uncertainty

# 获取PLL输出的50MHz系统时钟
set sys_clk [get_clocks *u_clk_pll*|*divclk]


# ==============================================
# 2. 核心算法模块内部 多周期约束
# 算法10kHz更新，5000个50MHz周期才执行1次
# ==============================================
set MULTICYCLE_NUM 5000

# 次级路径滤波多周期约束
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_s_hat_filter*|*] -to [get_registers *u_s_hat_filter*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_s_hat_filter*|*] -to [get_registers *u_s_hat_filter*|*]

# 谐波抑制多周期约束
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_harmonic_suppress*|*] -to [get_registers *u_harmonic_suppress*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_harmonic_suppress*|*] -to [get_registers *u_harmonic_suppress*|*]

# FLANN 顶层模块内部路径多周期约束
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_flann_top*|*] -to [get_registers *u_flann_top*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_flann_top*|*] -to [get_registers *u_flann_top*|*]

# Elman RNN 内部路径多周期约束
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_elman_rnn*|*] -to [get_registers *u_elman_rnn*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_elman_rnn*|*] -to [get_registers *u_elman_rnn*|*]

# 【重要恢复】FxLMS 核心内部路径多周期约束（因代码依然是全并行架构）
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_fxlms_core*|*] -to [get_registers *u_fxlms_core*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_fxlms_core*|*] -to [get_registers *u_fxlms_core*|*]


# ==============================================
# 3. 跨模块 接口多周期约束
# ==============================================
# ADC接口 -> 各算法模块的输入路径
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_adc_interface*|*] -to [get_registers *u_elman_rnn*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_adc_interface*|*] -to [get_registers *u_elman_rnn*|*]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_adc_interface*|*] -to [get_registers *u_flann_top*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_adc_interface*|*] -to [get_registers *u_flann_top*|*]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_adc_interface*|*] -to [get_registers *u_s_hat_filter*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_adc_interface*|*] -to [get_registers *u_s_hat_filter*|*]

# 【重要恢复】ADC接口 -> FxLMS 核心的输入路径
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_adc_interface*|*] -to [get_registers *u_fxlms_core*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_adc_interface*|*] -to [get_registers *u_fxlms_core*|*]

# 各算法模块 -> 顶层混合输出接口
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_flann_top*|*] -to [get_registers *]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_flann_top*|*] -to [get_registers *]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_elman_rnn*|*] -to [get_registers *]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_elman_rnn*|*] -to [get_registers *]

# 【重要恢复】FxLMS核心 -> 顶层混合输出接口
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_registers *u_fxlms_core*|*] -to [get_registers *]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_registers *u_fxlms_core*|*] -to [get_registers *]


# ==============================================
# 4. ADC/DAC接口约束（同步接口，匹配系统时钟）
# ==============================================
# ADC输入约束
set_input_delay -clock $sys_clk -max 2 [get_ports {adc_data* adc_data_valid}]
set_input_delay -clock $sys_clk -min 0.5 [get_ports {adc_data* adc_data_valid}]

# DAC输出约束
set_output_delay -clock $sys_clk -max 2 [get_ports {dac_data* dac_data_valid}]
set_output_delay -clock $sys_clk -min 0.5 [get_ports {dac_data* dac_data_valid}]


# ==============================================
# 5. 异步复位与静态配置虚假路径约束（不做时序检查）
# ==============================================
set_false_path -from [get_ports sys_rst_n] -to [all_registers]
set_false_path -from [get_cells *u_rst_sync*|*] -to [all_registers]

# 外部输入的静态配置信号，不需要满足高速时序要求
set_false_path -from [get_ports {flann_en rnn_en mix_mode[*] step_mode_sel harm_suppress_en}] -to [all_registers]
