`timescale 1ns/1ps
`default_nettype none

module npu_gemv_model_top #(
    parameter integer LANES=4,
    parameter integer MAX_FEATURES=64
) (
    input wire clk,rst,
    input wire activation_valid,
    output wire activation_ready,
    input wire signed [7:0] activation_data,
    input wire activation_last,
    input wire layer_cmd_valid,
    output wire layer_cmd_ready,
    input wire [15:0] layer_n,layer_k,
    input wire layer_relu,layer_last,
    input wire weight_valid,
    output wire weight_ready,
    input wire [LANES*8-1:0] weight_vector,
    input wire param_valid,
    output wire param_ready,
    input wire signed [31:0] param_bias,param_multiplier,
    input wire [5:0] param_shift,
    output wire busy,
    output reg layer_done,model_done,error,
    output wire result_valid,
    input wire result_ready,
    output wire [15:0] result_index,
    output wire signed [7:0] result_data,
    output wire result_last
);
    localparam [2:0] LOAD_INPUT=3'd0,DROP_INPUT=3'd1,WAIT_LAYER=3'd2,
                     ISSUE_LAYER=3'd3,RUN_LAYER=3'd4,DRAIN_RESULT=3'd5;
    reg [2:0] state;
    (* ram_style="distributed" *) reg signed [7:0] activation_bank0[0:MAX_FEATURES-1];
    (* ram_style="distributed" *) reg signed [7:0] activation_bank1[0:MAX_FEATURES-1];
    reg [15:0] activation_count,current_length,current_k;
    reg [15:0] pending_n,pending_k,drain_index,drain_length;
    reg current_bank,pending_relu,pending_last,result_bank;

    wire core_cmd_ready,core_operand_ready,core_param_ready,core_done,core_error;
    wire core_out_valid,core_out_last;
    wire [15:0] core_out_col;
    wire signed [7:0] core_out_data;
    wire signed [7:0] current_activation = current_bank
        ? activation_bank1[current_k] : activation_bank0[current_k];
    wire [15:0] pending_col = drain_index;
    wire signed [7:0] selected_result = result_bank
        ? activation_bank1[pending_col] : activation_bank0[pending_col];
    wire core_cmd_valid=(state==ISSUE_LAYER);
    wire core_operand_valid=(state==RUN_LAYER) && weight_valid;
    wire core_param_valid=(state==RUN_LAYER) && param_valid;
    wire core_out_ready=(state==RUN_LAYER);
    wire activation_fire=activation_valid && activation_ready;
    wire weight_fire=weight_valid && weight_ready;
    wire result_fire=result_valid && result_ready;

    assign activation_ready=!rst && (state==LOAD_INPUT || state==DROP_INPUT);
    assign layer_cmd_ready=!rst && state==WAIT_LAYER;
    assign weight_ready=!rst && state==RUN_LAYER && core_operand_ready;
    assign param_ready=!rst && state==RUN_LAYER && core_param_ready;
    assign busy=(state==DROP_INPUT || state==ISSUE_LAYER || state==RUN_LAYER || state==DRAIN_RESULT);
    assign result_valid=!rst && state==DRAIN_RESULT;
    assign result_index=drain_index;
    assign result_data=selected_result;
    assign result_last=result_valid && drain_index+16'd1==drain_length;

    npu_gemv_linear_top #(.LANES(LANES)) linear (
        .clk(clk),.rst(rst),.cmd_valid(core_cmd_valid),.cmd_ready(core_cmd_ready),
        .cmd_n(pending_n),.cmd_k(pending_k),.cmd_relu(pending_relu),
        .operand_valid(core_operand_valid),.operand_ready(core_operand_ready),
        .a_scalar(current_activation),.b_vector(weight_vector),
        .param_valid(core_param_valid),.param_ready(core_param_ready),
        .param_bias(param_bias),.param_multiplier(param_multiplier),.param_shift(param_shift),
        .busy(),.done(core_done),.error(core_error),.out_valid(core_out_valid),
        .out_ready(core_out_ready),.out_row(),.out_col(core_out_col),.out_data(core_out_data),
        .out_last(core_out_last),.stat_cycles(),.stat_load_cycles(),.stat_compute_cycles(),
        .stat_input_wait_cycles(),.stat_buffer_stall_cycles(),.stat_output_stall_cycles(),
        .stat_overlap_cycles(),.stat_mac_count()
    );

    always @(posedge clk) begin
        if(rst) begin
            state<=LOAD_INPUT;
            activation_count<=0;current_length<=0;current_k<=0;
            pending_n<=0;pending_k<=0;drain_index<=0;drain_length<=0;
            current_bank<=0;pending_relu<=0;pending_last<=0;result_bank<=0;
            layer_done<=0;model_done<=0;error<=0;
        end else begin
            layer_done<=0;model_done<=0;error<=0;

            if(state==LOAD_INPUT && activation_fire) begin
                if(activation_count==MAX_FEATURES-1 && !activation_last) begin
                    activation_count<=0;
                    state<=DROP_INPUT;
                    error<=1;
                end else begin
                    activation_bank0[activation_count]<=activation_data;
                    if(activation_last) begin
                        current_length<=activation_count+16'd1;
                        current_bank<=0;
                        activation_count<=0;
                        state<=WAIT_LAYER;
                    end else activation_count<=activation_count+1'b1;
                end
            end

            if(state==DROP_INPUT && activation_fire && activation_last) begin
                activation_count<=0;
                state<=LOAD_INPUT;
            end

            if(state==WAIT_LAYER && layer_cmd_valid && layer_cmd_ready) begin
                if(layer_n==0 || layer_k==0 || layer_k!=current_length || layer_n>MAX_FEATURES) begin
                    error<=1;
                end else begin
                    pending_n<=layer_n;
                    pending_k<=layer_k;
                    pending_relu<=layer_relu;
                    pending_last<=layer_last;
                    current_k<=0;
                    state<=ISSUE_LAYER;
                end
            end

            if(state==ISSUE_LAYER && core_cmd_ready) begin
                current_k<=0;
                state<=RUN_LAYER;
            end

            if(state==RUN_LAYER && weight_fire) begin
                if(current_k==pending_k-1'b1) current_k<=0;
                else current_k<=current_k+1'b1;
            end

            if(state==RUN_LAYER && core_out_valid && core_out_ready) begin
                if(current_bank)
                    activation_bank0[core_out_col]<=core_out_data;
                else activation_bank1[core_out_col]<=core_out_data;
            end

            if(state==RUN_LAYER && core_error) error<=1;
            if(state==RUN_LAYER && core_done) begin
                layer_done<=1;
                if(pending_last) begin
                    result_bank<=!current_bank;
                    drain_index<=0;
                    drain_length<=pending_n;
                    state<=DRAIN_RESULT;
                end else begin
                    current_bank<=!current_bank;
                    current_length<=pending_n;
                    state<=WAIT_LAYER;
                end
            end

            if(state==DRAIN_RESULT && result_fire) begin
                if(result_last) begin
                    drain_index<=0;
                    current_length<=0;
                    current_bank<=0;
                    activation_count<=0;
                    state<=LOAD_INPUT;
                    model_done<=1;
                end else drain_index<=drain_index+1'b1;
            end
        end
    end
endmodule

`default_nettype wire
