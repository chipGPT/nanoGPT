//The testbench for Pow2 unit
module Pow_tb();
    logic clk;
    logic signed [`DATA_SIZE-1:0] current_max;
    logic signed [`DATA_SIZE-1:0] input_vector;
    logic signed [`LARGE_SIZE:0] uSoftmax;
    integer write_file0, write_file1, write_file2;
    initial begin
            
        write_file0 = $fopen("Pow_data.txt", "w");
        write_file1 = $fopen("true_pow_data.txt", "w");
        write_file2 = $fopen("Pow_data2.txt", "w");
    end

    Pow2 Pow2 (
        .current_max,
        .input_vector,
        .uSoftmax //UnnormedSoftmax
    ); 
    int counter;
    real true_out, xj, uSoftmax_floating;
    always #5 clk = ~clk;

    //initialize "current_max", could also use rand()
    always #10 begin
        counter = counter + 1;
        if(counter == 5) begin
            current_max = 0;
            counter = 0;
        end
        else begin
            current_max = current_max+1;
        end
    end

    //initlalize input vectors
    always #5 begin
        input_vector[`DATA_SIZE-2:`FRAC] = input_vector[`DATA_SIZE-2:`FRAC] + 1;
        input_vector[`FRAC-1:0] = input_vector[`FRAC-1:0] + 1;
    end

    //calculate the expected value
    always #5 begin
        xj = input_vector[`DATA_SIZE-2:`FRAC] + (input_vector[`FRAC-1:0]/16.0);
        true_out = 2**(xj-current_max+0.0);
    end

    //transfer the output from fixed point 8 to a real number
    always #5 begin
        uSoftmax_floating = uSoftmax[`LARGE_SIZE-2:`FRAC] + uSoftmax[`FRAC-1:0]*1.0/16.0;
    end

    //write the outputs into output files
    always #5 begin
        $fdisplay(write_file0,"%0d\t",uSoftmax);
        $fdisplay(write_file1,"%0f\t",true_out);
        $fdisplay(write_file2,"%0f\t",uSoftmax_floating);
    end

    //testing begin
    initial begin
        clk = 0;
        counter = 0;
        current_max = 0;
        input_vector = 5;
        #200;
        $finish;
    end
endmodule