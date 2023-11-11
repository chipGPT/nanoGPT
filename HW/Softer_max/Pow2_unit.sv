//perform 2^(xj-localMax)
//To do: modify output data size
//To do: explore floating point 8 representation
module Pow2 (
    input logic signed [`DATA_SIZE-1:0] current_max,
    input logic signed [`DATA_SIZE-1:0] input_vector,
    output logic signed [`LARGE_SIZE:0] uSoftmax //UnnormedSoftmax
);  
    logic signed [`DATA_SIZE-1:0] pow2_frac;
    logic signed [`LARGE_SIZE-1:0] FP_1;

    //fixed_point 2
    always_comb begin
        FP_1 = 0;
        FP_1[`FRAC] = 1;
    end

    //return [2^(n/16)]*2^4 and then round to integer
    LUT LUT0 (
        .index(input_vector[`FRAC-1:0]),
        .out(pow2_frac)
    );

    //for debug
    logic signed [`DATA_SIZE-1-`FRAC:0] temp;
    assign temp = input_vector [`DATA_SIZE-1:`FRAC];
    //2^[(int xj)-localMax]*2^(frac+4)/2^4
    //8'b01000000, 2 in FixedP8
    assign uSoftmax = ((FP_1 <<< (temp - current_max)) * pow2_frac) >>> `FRAC;
endmodule