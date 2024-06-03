module test();

    localparam half_cycle = 5;
    localparam delay = 0.2;

    reg clk;
    reg [31:0] x;
    wire [31:0] y;

    silu #(
        .I_EXP(8),
        .I_MNT(23)
    ) dut0 (
        .idata(x),
        .odata(y)
    );

    always begin
        #half_cycle clk = ~clk;
    end

    initial begin

//        `ifdef SYNTH
//            $sdf_annotate("fadd.sdf", fp32add,,,"MAXIMUM");
//        `endif

        $dumpfile("silu.vcd");
        $dumpvars;

//        $readmemh("test.mem", mem);

//        fd = $fopen("fadd_sv_result.txt", "w"); 

        clk = 0;

        x = 32'hC040_0000;

        repeat (10) begin
            @(posedge clk); #delay;
            x = $random();
        end

        $finish;

    end


endmodule

