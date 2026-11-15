`timescale 1ns/1ps

module tb_npu_gemv_model #(
    parameter integer LANES=2,MAX_FEATURES=64,STALLS=1
);
    localparam MAX_LAYERS=64;
    reg clk=0;always #5 clk=~clk;
    reg rst=1;
    reg activation_valid=0,activation_last=0;
    wire activation_ready;
    reg signed [7:0] activation_data=0;
    reg layer_cmd_valid=0,layer_relu=0,layer_last=0;
    wire layer_cmd_ready;
    reg [15:0] layer_n=0,layer_k=0;
    reg weight_valid=0;
    wire weight_ready;
    reg [LANES*8-1:0] weight_vector=0;
    reg param_valid=0;
    wire param_ready;
    reg signed [31:0] param_bias=0,param_multiplier=1;
    reg [5:0] param_shift=0;
    wire busy,layer_done,model_done,error;
    wire result_valid,result_last;
    reg result_ready=0;
    wire [15:0] result_index;
    wire signed [7:0] result_data;

    npu_gemv_model_top #(.LANES(LANES),.MAX_FEATURES(MAX_FEATURES)) dut (
        .clk(clk),.rst(rst),
        .activation_valid(activation_valid),.activation_ready(activation_ready),
        .activation_data(activation_data),.activation_last(activation_last),
        .layer_cmd_valid(layer_cmd_valid),.layer_cmd_ready(layer_cmd_ready),
        .layer_n(layer_n),.layer_k(layer_k),.layer_relu(layer_relu),.layer_last(layer_last),
        .weight_valid(weight_valid),.weight_ready(weight_ready),.weight_vector(weight_vector),
        .param_valid(param_valid),.param_ready(param_ready),.param_bias(param_bias),
        .param_multiplier(param_multiplier),.param_shift(param_shift),
        .busy(busy),.layer_done(layer_done),.model_done(model_done),.error(error),
        .result_valid(result_valid),.result_ready(result_ready),.result_index(result_index),
        .result_data(result_data),.result_last(result_last)
    );

    integer jobs_fd,input_fd,weights_fd,params_fd,expected_fd,results_fd;
    integer job_count,file_lanes,m,n,k,relu,value,c,job,loaded,final_count;
    integer job_n[0:MAX_LAYERS-1],job_k[0:MAX_LAYERS-1],job_relu[0:MAX_LAYERS-1];
    integer final_expected[0:MAX_FEATURES-1];
    integer initial_values[0:MAX_FEATURES-1];
    integer result_count=0,error_count=0,layer_count=0,model_count=0;
    integer old_errors,old_layers,stalled_index=-1,stalled_cycles=0,output_stall_cycles=0;
    string directory,marker,token;
    reg output_checking=0,stalled=0;
    reg [24:0] held_result;

    always @(negedge clk) begin
        if(!output_checking) result_ready=0;
        else if(!STALLS) result_ready=1;
        else if(result_valid && result_index!=stalled_index) begin
            stalled_index=result_index;
            stalled_cycles=0;
            result_ready=0;
        end else if(result_valid && stalled_cycles<2) begin
            stalled_cycles=stalled_cycles+1;
            result_ready=0;
        end else result_ready=1;
    end

    always @(posedge clk) if(!rst) begin
        if(stalled&&(!result_valid||{result_last,result_index,result_data}!==held_result))
            $fatal(1,"model result changed during backpressure");
        stalled=result_valid&&!result_ready;
        if(stalled) output_stall_cycles=output_stall_cycles+1;
        held_result={result_last,result_index,result_data};
        if(error) error_count=error_count+1;
        if(layer_done) layer_count=layer_count+1;
        if(model_done) model_count=model_count+1;
        if(result_valid&&result_ready) begin
            if(!output_checking) $fatal(1,"intermediate activation escaped to the host");
            if(result_index!=result_count||result_count>=final_count)
                $fatal(1,"model result index/count invalid");
            if($signed(result_data)!==final_expected[result_count])
                $fatal(1,"model result %0d got %0d expected %0d",result_count,
                       $signed(result_data),final_expected[result_count]);
            if(result_last!==(result_count==final_count-1)) $fatal(1,"model result last mismatch");
            $fdisplay(results_fd,"%0d",$signed(result_data));
            result_count=result_count+1;
        end
    end

    task automatic read_decimal(input integer fd,input integer minimum,maximum,
                                input string label_text,output integer parsed);
        integer scanned;
        begin
            scanned=$fscanf(fd,"%d",parsed);
            if(scanned!=1||parsed<minimum||parsed>maximum)
                $fatal(1,"model %s missing or outside range",label_text);
        end
    endtask

    task automatic require_eof(input integer fd,input string label_text);
        integer scanned;
        reg [1023:0] extra;
        begin
            scanned=$fscanf(fd,"%s",extra);
            if(scanned>0) $fatal(1,"model %s has extra tokens",label_text);
        end
    endtask

    task automatic send_activation(input integer count,input integer mark_last);
        integer index;
        begin
            for(index=0;index<count;index=index+1) begin
                @(negedge clk);
                activation_data=index;
                activation_last=mark_last&&(index==count-1);
                activation_valid=1;
                do @(posedge clk);while(!activation_ready);
                @(negedge clk);activation_valid=0;activation_last=0;
            end
        end
    endtask

    task automatic send_initial(input integer count);
        integer index;
        begin
            for(index=0;index<count;index=index+1) begin
                @(negedge clk);
                activation_data=initial_values[index];
                activation_last=(index==count-1);
                activation_valid=1;
                do @(posedge clk);while(!activation_ready);
                @(negedge clk);activation_valid=0;activation_last=0;
            end
        end
    endtask

    task automatic send_invalid_command(input integer invalid_n,input integer invalid_k);
        begin
            old_errors=error_count;
            @(negedge clk);layer_n=invalid_n;layer_k=invalid_k;
            layer_relu=0;layer_last=0;layer_cmd_valid=1;
            do @(posedge clk);while(!layer_cmd_ready);
            @(negedge clk);layer_cmd_valid=0;
            while(error_count==old_errors) @(negedge clk);
            if(busy||!layer_cmd_ready) $fatal(1,"invalid model layer changed controller state");
        end
    endtask

    task automatic send_weights(input integer cols,input integer depth);
        integer base,inner,scanned;
        begin
            for(base=0;base<cols;base=base+LANES)
                for(inner=0;inner<depth;inner=inner+1) begin
                    if($fscanf(weights_fd,"%s",token)!=1) $fatal(1,"model weights truncated");
                    if(token.len()!=2*LANES||$sscanf(token,"%h",weight_vector)!=1)
                        $fatal(1,"model weight encoding invalid");
                    if(STALLS&&(base+inner)%5==2) repeat(2) @(negedge clk);
                    @(negedge clk);weight_valid=1;
                    do @(posedge clk);while(!weight_ready);
                    @(negedge clk);weight_valid=0;
                end
            scanned=0;
        end
    endtask

    task automatic send_parameters(input integer cols);
        integer column,pr,pc,bias,multiplier,shift;
        begin
            for(column=0;column<cols;column=column+1) begin
                read_decimal(params_fd,0,0,"parameter row",pr);
                read_decimal(params_fd,column,column,"parameter column",pc);
                read_decimal(params_fd,-2147483647-1,2147483647,"parameter bias",bias);
                read_decimal(params_fd,1,2147483647,"parameter multiplier",multiplier);
                read_decimal(params_fd,0,31,"parameter shift",shift);
                param_bias=bias;param_multiplier=multiplier;param_shift=shift;
                if(STALLS&&column%4==1) repeat(2) @(negedge clk);
                @(negedge clk);param_valid=1;
                do @(posedge clk);while(!param_ready);
                @(negedge clk);param_valid=0;
            end
        end
    endtask

    initial begin
        #2000000 $fatal(1,"model timeout");
    end

    initial begin
        if(!$value$plusargs("BUNDLE=%s",directory)) $fatal(1,"pass +BUNDLE=directory");
        jobs_fd=$fopen({directory,"/jobs.txt"},"r");
        input_fd=$fopen({directory,"/inputs.txt"},"r");
        weights_fd=$fopen({directory,"/model-weights.txt"},"r");
        params_fd=$fopen({directory,"/parameters.txt"},"r");
        expected_fd=$fopen({directory,"/expected.txt"},"r");
        results_fd=$fopen({directory,"/rtl-model-results.txt"},"w");
        if(!jobs_fd||!input_fd||!weights_fd||!params_fd||!expected_fd||!results_fd)
            $fatal(1,"model bundle file unavailable");
        if($fscanf(jobs_fd,"%s %d %d",marker,job_count,file_lanes)!=3||marker!="gemv-v1"
                ||job_count<1||job_count>MAX_LAYERS||file_lanes!=LANES)
            $fatal(1,"model GEMV header/config invalid");
        for(job=0;job<job_count;job=job+1) begin
            read_decimal(jobs_fd,1,1,"batch",m);
            read_decimal(jobs_fd,1,MAX_FEATURES,"output width",n);
            read_decimal(jobs_fd,1,MAX_FEATURES,"input width",k);
            read_decimal(jobs_fd,0,1,"relu",relu);
            if(job>0&&k!=job_n[job-1]) $fatal(1,"model layer dimensions do not chain");
            job_n[job]=n;job_k[job]=k;job_relu[job]=relu;
            for(c=0;c<n;c=c+1) begin
                read_decimal(expected_fd,-128,127,"expected output",value);
                if(job==job_count-1) final_expected[c]=value;
            end
            if(job==job_count-1) final_count=n;
        end
        require_eof(jobs_fd,"jobs");require_eof(expected_fd,"expected outputs");
        for(c=0;c<job_k[0];c=c+1) begin
            read_decimal(input_fd,-128,127,"initial activation",initial_values[c]);
        end
        loaded=job_k[0];
        require_eof(input_fd,"initial activations");
        repeat(4) @(negedge clk);rst=0;
        send_initial(loaded);
        if(!layer_cmd_ready) $fatal(1,"model did not finish loading initial activations");

        send_invalid_command(1,job_k[0]+1);
        send_invalid_command(0,job_k[0]);
        send_invalid_command(MAX_FEATURES+1,job_k[0]);

        for(job=0;job<job_count;job=job+1) begin
            old_layers=layer_count;
            @(negedge clk);
            layer_n=job_n[job];layer_k=job_k[job];layer_relu=job_relu[job];
            layer_last=(job==job_count-1);layer_cmd_valid=1;
            do @(posedge clk);while(!layer_cmd_ready);
            @(negedge clk);layer_cmd_valid=0;
            fork
                send_weights(job_n[job],job_k[job]);
                send_parameters(job_n[job]);
            join
            while(layer_count==old_layers) @(negedge clk);
            if(job==job_count-1) output_checking=1;
        end
        while(result_count<final_count||model_count==0) @(negedge clk);
        if(result_count!=final_count||model_count!=1||layer_count!=job_count)
            $fatal(1,"model output/completion count mismatch");
        if(STALLS&&output_stall_cycles<2*final_count)
            $fatal(1,"model output backpressure was not exercised");
        require_eof(weights_fd,"weight stream");require_eof(params_fd,"parameter stream");
        $display("PASS tb_npu_gemv_model LANES=%0d layers=%0d",LANES,job_count);

        old_errors=error_count;
        send_activation(MAX_FEATURES+1,1);
        if(error_count!=old_errors+1||!activation_ready)
            $fatal(1,"overlong activation did not report capacity error and recover");
        send_activation(1,0);
        @(negedge clk);rst=1;
        repeat(2) @(negedge clk);
        rst=0;
        send_activation(1,1);
        if(!layer_cmd_ready) $fatal(1,"model reset did not recover input loading");
        $fclose(results_fd);$finish;
    end
endmodule
