
`ifndef __RELU_V_
`define __RELU_V_

module relu #(
    parameter   I_EXP   = 8,
    parameter   I_MNT   = 23,
    parameter   I_DATA  = I_EXP + I_MNT + 1
)(
    input       [I_DATA-1:0]    idata,
    output reg  [I_DATA-1:0]    odata
);

    always @(*) begin
        odata = (idata[I_DATA-1])? I_DATA'h0: idata;
    end

endmodule

`endif

