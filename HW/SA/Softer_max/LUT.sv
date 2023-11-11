// a LUT used for pow2 unit, will store this LUT into a on-chip mem in the future
// this LUT contains value from 2^(0/16) to 2^(15/16)
module LUT(
    input logic signed [`FRAC-1:0] index,
    output logic signed [`DATA_SIZE-1:0] out
);
    always_comb begin
        //[2^(n/16)] and then round to integer
        out = 8'b0;
        case(index)
            4'b0000 : out = 8'b00010000;
            4'b0001 : out = 8'b00010001;
            4'b0010 : out = 8'b00010001;
            4'b0011 : out = 8'b00010010;
            4'b0100 : out = 8'b00010011;
            4'b0101 : out = 8'b00010100;
            4'b0110 : out = 8'b00010101;
            4'b0111 : out = 8'b00010110;
            4'b1000 : out = 8'b00010111;
            4'b1001 : out = 8'b00011000;
            4'b1010 : out = 8'b00011001;
            4'b1011 : out = 8'b00011010;
            4'b1100 : out = 8'b00011011;
            4'b1101 : out = 8'b00011100;
            4'b1110 : out = 8'b00011101;
            4'b1111 : out = 8'b00011111;          
        endcase
    end
endmodule