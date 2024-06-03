// 12/04/2024 Ruichen Qi changed, delay logic removed  
module counter
#(
    parameter MAX_COUNT=31,
    localparam BIT_WIDTH=$clog2(MAX_COUNT+1)
)
(
    input clk,
    input rstn,
    input inc,
    //input new_inst,
    output overflow,
    output logic [BIT_WIDTH-1:0] out
);
// logic inc_d;
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        out <= 0;
//      inc_d <= 0;
    end
    else begin
//      inc_d <= inc;
	    //if ((inc&&overflow) || new_inst) out<=0;
        if ((inc && overflow) ) out <= 0;
	    else if (inc) out <= out+1;
	    else out <= out;
    end
end

// assign overflow=(out==MAX_COUNT) & inc_d;
assign overflow=(out==MAX_COUNT) & inc;

endmodule