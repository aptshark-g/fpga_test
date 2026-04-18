// 异步复位、同步释放模块
// 解决异步复位的亚稳态问题，消除全芯片hold违例
`timescale 1ns / 1ps

module rst_sync(
    input           sys_clk,        // 系统全局时钟
    input           sys_rst_n,      // 外部输入异步复位，低电平有效
    output  reg     sys_rst_n_sync  // 同步后复位，全芯片使用，低电平有效
);

reg rst_n_r1;

// 两级寄存器同步，异步复位，同步释放
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        rst_n_r1      <= 1'b0;
        sys_rst_n_sync<= 1'b0;
    end else begin
        rst_n_r1      <= 1'b1;
        sys_rst_n_sync<= rst_n_r1;
    end
end

endmodule