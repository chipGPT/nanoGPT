`ifndef __FDIV2_V_
`define __FDIV2_V_

module fdiv2 #(
    parameter   I_EXP   = 8,
    parameter   I_MNT   = 7,
    parameter   I_DATA  = I_EXP + I_MNT + 1
)(
    input       [I_DATA-1:0]    if32,
    output reg  [I_DATA-1:0]    of32
);

    wire d_SGN;
    wire [I_EXP-1:0] d_EXP;
    wire [I_MNT-1:0] d_MAT;
    assign d_SGN = if32[I_EXP+I_MNT];
    assign d_EXP = if32[I_EXP+I_MNT-1:I_MNT];
    assign d_MAT = if32[I_MNT-1:0];

    always @(*) begin
        case(d_EXP)
            {I_EXP{1'b0}}: begin                // shift MAT
                of32 = {d_SGN, d_EXP, {1'b0, d_MAT[I_MNT-1:1]}};
            end
            {{(I_EXP-1){1'b0}}, 1'b1}: begin    // spec case 1
                of32 = {d_SGN, {I_EXP{1'b0}}, {1'b1, d_MAT[I_MNT-1:1]}};
            end
            default: begin
                of32 = {d_SGN, d_EXP-1'b1, d_MAT};
            end
        endcase
    end

endmodule

`endif

