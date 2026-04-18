# ==============================================
# 1. 主时钟与PLL约束（核心时钟定义）
# ==============================================
# 外部输入50MHz参考时钟
create_clock -name sys_clk_in -period 20 -waveform {0 10} [get_ports sys_clk]
# 自动推导PLL输出时钟，继承主时钟约束
derive_pll_clocks -create_base_clocks
# 自动推导时钟不确定性，预留时序余量
derive_clock_uncertainty

# 获取PLL输出的50MHz系统时钟（工具自动生成的时钟名，无需手动改）
set sys_clk [get_clocks *u_clk_pll*|*divclk]

# ==============================================
# 2. 核心多周期约束（解决时序违例的关键！）
# ==============================================
# 你的算法10kHz更新，5000个50MHz周期才执行1次，给5000个周期setup余量
set MULTICYCLE_NUM 5000

# 对所有算法模块的内部路径，设置多周期约束（完全匹配sample_en使能逻辑）
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_cells *u_fxlms_core*|*] -to [get_cells *u_fxlms_core*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_cells *u_fxlms_core*|*] -to [get_cells *u_fxlms_core*|*]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_cells *u_s_hat_filter*|*] -to [get_cells *u_s_hat_filter*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_cells *u_s_hat_filter*|*] -to [get_cells *u_s_hat_filter*|*]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_cells *u_harmonic_suppress*|*] -to [get_cells *u_harmonic_suppress*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_cells *u_harmonic_suppress*|*] -to [get_cells *u_harmonic_suppress*|*]

# 接口模块多周期约束
set_multicycle_path -setup $MULTICYCLE_NUM -from [get_cells *u_adc_interface*|*] -to [get_cells *u_fxlms_core*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_cells *u_adc_interface*|*] -to [get_cells *u_fxlms_core*|*]

set_multicycle_path -setup $MULTICYCLE_NUM -from [get_cells *u_fxlms_core*|*] -to [get_cells *u_dac_interface*|*]
set_multicycle_path -hold  [expr $MULTICYCLE_NUM - 1] -from [get_cells *u_fxlms_core*|*] -to [get_cells *u_dac_interface*|*]

# ==============================================
# 3. ADC/DAC接口约束（同步接口，匹配系统时钟）
# ==============================================
# ADC输入约束
set_input_delay -clock $sys_clk -max 2 [get_ports {adc_data* adc_data_valid}]
set_input_delay -clock $sys_clk -min 0.5 [get_ports {adc_data* adc_data_valid}]

# DAC输出约束
set_output_delay -clock $sys_clk -max 2 [get_ports {dac_data* dac_data_valid}]
set_output_delay -clock $sys_clk -min 0.5 [get_ports {dac_data* dac_data_valid}]

# ==============================================
# 4. 异步复位虚假路径约束（不做时序检查）
# ==============================================
set_false_path -from [get_ports sys_rst_n] -to [all_registers]
set_false_path -from [get_cells *u_rst_sync*|*] -to [all_registers]

# ==============================================
# 5. 关闭不必要的优化，大幅缩短编译时间
# ==============================================
set_global_assignment -name OPTIMIZATION_MODE "BALANCED"
set_global_assignment -name FITTER_EFFORT "STANDARD FIT"
set_global_assignment -name PHYSICAL_SYNTHESIS_EFFORT "NORMAL"
set_global_assignment -name AUTO_DELAY_CHAINS_FOR_HOLD_TIMING ON
set_global_assignment -name ENABLE_HOLD_BACK_OFF OFF